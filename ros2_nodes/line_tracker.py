#!/usr/bin/env python3
"""하향 카메라 이진화 기반 구획선(라인) 검출 로직.

search_phase_node.py의 비행 제어 로직과 분리해서, C270 같은 일반 웹캠으로도
(ROS2/PX4 없이) 곧바로 검출 품질을 확인할 수 있게 뺐다. 실제 사용처는
search_phase_node.py이고, test_line_tracker_camera.py는 이 모듈을 그대로
가져다 눈으로 확인하는 용도.
"""
import math

import cv2
import numpy as np


class LineTrackResult:
    __slots__ = ("line_found", "e_y", "e_psi", "intersection")

    def __init__(self, line_found, e_y, e_psi, intersection):
        self.line_found = line_found
        self.e_y = e_y            # 횡방향 편차(px, 화면 중앙 기준, +면 라인이 오른쪽)
        self.e_psi = e_psi        # 각도 편차(rad, ROI 상단/하단 중심 offset으로 추정)
        self.intersection = intersection  # 교차점(십자) 검출 여부


class LineTracker:
    """하향 카메라 이진화 기반 라인트레이싱. 격자선이 바닥과 대비되는 색이라는
    규정 전제를 그대로 이용 — Otsu 이진화로 색상 확정 없이도 라인 대 바닥을
    분리할 수 있게 함."""

    def __init__(self, roi_top_ratio=0.55, roi_bottom_ratio=0.95, cross_row_ratio=0.5):
        self.roi_top_ratio = roi_top_ratio
        self.roi_bottom_ratio = roi_bottom_ratio
        self.cross_row_ratio = cross_row_ratio  # 교차점 판정용 가로 스캔 위치(ROI 내부)
        # 직전 프레임에서 찾은 위치 - 연속성 판단용(2026-09-07 실측: 발처럼
        # 라인과 비슷한 굵기의 물체가 화면에 새로 나타나면 그쪽으로 튀는
        # 문제가 있어서, "이전 위치와 가까운 쪽"을 우선하도록 함).
        self._last_bottom_x = None
        self._last_top_x = None
        # 두께 가중 평활화(EMA) 상태 - 얇고(신뢰도 낮은) 검출일수록 출력을
        # 덜 바꾸고, 두꺼울수록(신뢰도 높을수록) 빨리 반영해서 흔들림/노이즈
        # 로 인한 e_y·e_psi 떨림을 줄인다.
        self._smoothed_e_y = None
        self._smoothed_e_psi = None

    @staticmethod
    def _binarize(gray):
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        # 라인이 바닥보다 밝을 수도 어두울 수도 있어서, 흰 픽셀이 더 적은 쪽을
        # "라인"으로 취급 (라인 폭 10cm << 바닥 면적이라는 전제).
        if cv2.countNonZero(mask) > mask.size // 2:
            mask = cv2.bitwise_not(mask)
        return mask

    def _row_centroid(self, mask_row, prefer_x=None):
        """반환값: (중심 x, 폭 px) 또는 못 찾으면 None."""
        xs = np.nonzero(mask_row)[0]
        if xs.size == 0:
            return None
        # 전체 픽셀 평균 대신, 가장 넓은 "연속된" 구간의 중심을 쓴다.
        # 평균은 바닥 텍스처/그라우트선 같은 잡음이 섞이면 실제 라인이
        # 아니라 잡음 쪽으로 중심이 쏠리는 문제가 실측됨(2026-09-07,
        # 화강암 타일 바닥에서 검은 테이프 대신 타일 이음새에 점이
        # 찍힘). 라인은 화면에서 가장 굵은 연속 어두운/밝은 구간이라는
        # 전제가 더 안전하다.
        #
        # 다만 "무조건 가장 넓은 구간"만 고르면, 렌즈 비네팅/그림자처럼
        # 라인보다 훨씬 넓은 어두운 덩어리가 있을 때 그쪽을 잘못 고르는
        # 문제가 또 실측됨(2026-09-07, 같은 세션) -> 라인이라고 믿을만한
        # 폭 상한을 두고, 그 상한을 넘는 구간만 있으면 차라리 "못 찾음"
        # 으로 처리한다(엉뚱한 위치를 자신있게 보고하는 것보다 안전).
        gaps = np.where(np.diff(xs) > 1)[0]
        starts = np.concatenate(([0], gaps + 1))
        ends = np.concatenate((gaps, [len(xs) - 1]))
        widths = xs[ends] - xs[starts] + 1

        max_reasonable_width = max(20, int(0.15 * len(mask_row)))
        valid = widths <= max_reasonable_width
        if not np.any(valid):
            return None

        valid_idx = np.where(valid)[0]
        if prefer_x is not None:
            # 발처럼 라인과 비슷한 굵기의 물체가 화면에 새로 나타나면
            # (2026-09-07 실측) 폭 기준만으로는 못 걸러서, 후보가 여럿이면
            # 직전 프레임 위치와 가장 가까운 쪽을 우선한다(연속성 가정 -
            # 라인은 프레임 사이 위치가 크게 안 튐).
            centers = (xs[starts[valid_idx]] + xs[ends[valid_idx]]) / 2.0
            best = valid_idx[int(np.argmin(np.abs(centers - prefer_x)))]
        else:
            best = valid_idx[np.argmax(widths[valid_idx])]
        run = xs[starts[best]:ends[best] + 1]
        width = int(run[-1] - run[0] + 1)
        return float(run.mean()), width

    def process(self, bgr_image) -> LineTrackResult:
        h, w = bgr_image.shape[:2]
        y0, y1 = int(h * self.roi_top_ratio), int(h * self.roi_bottom_ratio)
        roi = bgr_image[y0:y1, :]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        mask = self._binarize(gray)

        # 세로로 위/아래 두 지점의 라인 무게중심을 구해서 e_y(하단 기준)와
        # e_psi(위/아래 무게중심 차이로 추정한 각도)를 동시에 뽑는다.
        bottom_result = self._row_centroid(mask[-5, :] if mask.shape[0] > 5 else mask[-1, :],
                                            prefer_x=self._last_bottom_x)
        top_result = self._row_centroid(mask[5, :] if mask.shape[0] > 5 else mask[0, :],
                                         prefer_x=self._last_top_x)

        if bottom_result is None:
            self._last_bottom_x = None
            self._last_top_x = None
            self._smoothed_e_y = None
            self._smoothed_e_psi = None
            return LineTrackResult(False, 0.0, 0.0, False)

        bottom_c, bottom_width = bottom_result
        top_c, _top_width = top_result if top_result is not None else (None, 0)

        self._last_bottom_x = bottom_c
        self._last_top_x = top_c

        raw_e_y = bottom_c - (w / 2.0)
        if top_c is not None:
            dy_px = max(1, mask.shape[0] - 10)
            raw_e_psi = math.atan2(top_c - bottom_c, dy_px)
        else:
            raw_e_psi = 0.0

        # 두께 가중 평활화(EMA): 두꺼울수록(신뢰도 높음) 새 값을 빨리
        # 반영(alpha 큼), 얇을수록(신뢰도 낮음) 기존 값을 더 유지해서
        # 노이즈성 검출로 인한 떨림을 줄인다. alpha 범위 0.2~0.8.
        max_reasonable_width = max(20, int(0.15 * w))
        confidence = min(1.0, bottom_width / max_reasonable_width)
        alpha = 0.2 + 0.6 * confidence

        if self._smoothed_e_y is None:
            self._smoothed_e_y = raw_e_y
            self._smoothed_e_psi = raw_e_psi
        else:
            self._smoothed_e_y = alpha * raw_e_y + (1 - alpha) * self._smoothed_e_y
            self._smoothed_e_psi = alpha * raw_e_psi + (1 - alpha) * self._smoothed_e_psi

        intersection = self._detect_intersection(mask)
        return LineTrackResult(True, self._smoothed_e_y, self._smoothed_e_psi, intersection)

    def _detect_intersection(self, mask):
        """ROI 중간 높이에서 가로로 스캔했을 때 라인 폭이 정상 구획선 폭보다
        훨씬 넓게(또는 라인 구간이 2개 이상) 잡히면 교차점(십자)으로 판정.
        정상 라인트레이싱 중에는 세로선 하나만 폭 좁게 잡혀야 함."""
        row_idx = int(mask.shape[0] * self.cross_row_ratio)
        row = mask[row_idx, :]
        xs = np.nonzero(row)[0]
        if xs.size == 0:
            return False
        span = xs.max() - xs.min()
        # ROI 폭의 60% 이상을 라인이 덮으면 가로줄과 겹친 교차점으로 판단.
        return span > 0.6 * mask.shape[1]

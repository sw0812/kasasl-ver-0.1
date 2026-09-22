#!/usr/bin/env python3
"""KASA 중급부문 실내조난자 탐색임무 — 탐색 페이즈(1단계) 노드.

world 좌표계는 버티포트 = (0,0,0) (2026-08-18 확정, gz model -p 실측 검증됨).
Mission area 내부 격자 교차점(7x6=28개 중 후보) 좌표는
~/Downloads/kasa_gazebo_world/models/kasa_ground_field/waypoints.json의
all_grid_intersections_28과 동일한 값을 그대로 씀 — x∈{5,8,11,14,17,20,23},
y∈{5,8,11,14}. (요청 원문의 x∈[-9,9], y∈[-4.5,4.5]는 Mission area 중심을
원점으로 뒀을 때의 좌표라 버티포트=(0,0,0) 확정치와 안 맞아서 world 좌표로
바꿔서 씀 — 두 좌표계의 차이는 정확히 Mission area 중심 오프셋 (14, 9.5).)

파이프라인에 필요한 실제 라인트레이싱/그리드서치 코드가 기존 ros2_ws에
없어서(grep 확인 완료) 새로 작성. PX4 제어 패턴(QoS, 토픽 버전 접미사,
OffboardControlMode/TrajectorySetpoint 사용법)은 precision_landing/landing.py
컨벤션을 그대로 따름 — 서로 다른 노드가 다른 QoS/토픽 이름을 쓰면 나중에
디버깅하기 괴로워지므로.
"""
import collections
import enum
import math
import statistics

import cv2
import numpy as np
import rclpy
from line_tracker import LineTrackResult, LineTracker
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from aruco_opencv_msgs.msg import ArucoDetection
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleLocalPosition,
)

# ===========================================================================
# 1) 격자 정의 — world 좌표 (버티포트 원점 기준, 확정치)
# ===========================================================================
X_VALUES = [5.0, 8.0, 11.0, 14.0, 17.0, 20.0, 23.0]   # i = 0..6 (열)
Y_VALUES = [5.0, 8.0, 11.0, 14.0]                       # j = 0..3 (행)
# 경로점 개수는 규정상 고정값이 아니라 "대회 당일 현장에서 통보 -> 비행 전
# 참가팀이 시스템에 입력"하는 값이라 상수로 박아두지 않고 ROS 파라미터로 뺌
# (기본값 4는 규정집 예시 수치일 뿐, 실행 시 --ros-args -p n_markers_expected:=N
# 로 덮어쓰는 걸 전제).
DEFAULT_N_MARKERS_EXPECTED = 4
VERTIPORT_MARKER_ID = 0   # waypoints.json: vertiport.center_marker_id — 미션 경로점 아님


class Heading(enum.IntEnum):
    """ENU 기준 나침반 방위. 값은 시계 방향 순서라 (target-current)%4로
    좌/우/직진/유턴을 그대로 계산할 수 있게 잡음."""
    N = 0
    E = 1
    S = 2
    W = 3


_HEADING_VECTOR = {
    Heading.N: (0, 1),
    Heading.E: (1, 0),
    Heading.S: (0, -1),
    Heading.W: (-1, 0),
}

# 각 방위가 가리키는 절대 yaw(rad, ENU: E=0, 반시계+) — 교차점에서 실제로
# 이 각도까지 회전을 완료시키는 데 씀 (_HEADING_VECTOR와 정의가 어긋나지 않도록
# 벡터에서 직접 계산).
_HEADING_YAW = {h: math.atan2(dy, dx) for h, (dx, dy) in _HEADING_VECTOR.items()}


# ===========================================================================
# 2) 순수 함수 — ROS 의존성 없음, 단위테스트 가능
# ===========================================================================

def zigzag_order(n_cols: int, n_rows: int):
    """지그재그(boustrophedon) 방문 순서. 짝수 행(j 0,2,...)은 i 증가,
    홀수 행은 i 감소. 반환값은 (i, j) 튜플 리스트, 길이 n_cols*n_rows."""
    order = []
    for j in range(n_rows):
        cols = range(n_cols) if j % 2 == 0 else range(n_cols - 1, -1, -1)
        for i in cols:
            order.append((i, j))
    return order


class TurnCmd(enum.Enum):
    STRAIGHT = "STRAIGHT"
    LEFT = "LEFT"
    RIGHT = "RIGHT"
    UTURN = "UTURN"


Command = collections.namedtuple("Command", ["turn", "heading", "target_i", "target_j"])


def commands_from_order(order, start_heading=Heading.E):
    """방문 순서를 (교차점 도달 시 실행할 회전 명령) 큐로 변환.

    각 Command는 "이 교차점에 도달했을 때, 다음 교차점으로 가려면 어떤 회전을
    해야 하는가"를 뜻함 — 그래서 길이는 len(order)-1 (마지막 지점엔 다음이 없음).
    """
    commands = collections.deque()
    heading = start_heading
    for (i0, j0), (i1, j1) in zip(order, order[1:]):
        dx, dy = i1 - i0, j1 - j0
        needed = next(h for h, v in _HEADING_VECTOR.items() if v == (
            1 if dx > 0 else -1 if dx < 0 else 0,
            1 if dy > 0 else -1 if dy < 0 else 0,
        ))
        diff = (needed - heading) % 4
        turn = {0: TurnCmd.STRAIGHT, 1: TurnCmd.RIGHT, 2: TurnCmd.UTURN, 3: TurnCmd.LEFT}[diff]
        commands.append(Command(turn, needed, i1, j1))
        heading = needed
    return commands


def grid_xy(i, j):
    return X_VALUES[i], Y_VALUES[j]


def expand_axis_path(start, end):
    """두 격자 셀 사이를 인접한 한 칸씩만 잇는 중간 경로로 분해(i축 먼저,
    그다음 j축). commands_from_order/_HEADING_VECTOR는 4방위 인접 이동만
    표현 가능한데, 마커 방문 순서(ID 오름차순)는 인접하지 않거나 대각선인
    두 칸을 바로 이어야 할 수 있어서 — 그대로 넘기면 _HEADING_VECTOR에
    없는 대각선 방향을 찾다가 StopIteration이 난다(실비행 테스트로 확인됨:
    ID 순서상 (2,0)->(1,3) 같은 대각선 전환에서 재현)."""
    (i0, j0), (i1, j1) = start, end
    path = [(i0, j0)]
    i, j = i0, j0
    step_i = 1 if i1 > i else -1
    while i != i1:
        i += step_i
        path.append((i, j))
    step_j = 1 if j1 > j else -1
    while j != j1:
        j += step_j
        path.append((i, j))
    return path


# ===========================================================================
# 3) 라인트레이싱 비전 파이프라인 — ROI 크롭 -> 이진화 -> 무게중심 -> e_y/e_psi
#    + 교차점(십자 패턴) 검출 (구현은 line_tracker.py, 위에서 import)
# ===========================================================================


# ===========================================================================
# 4) ROS2 노드
# ===========================================================================

PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)
IMAGE_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    history=HistoryPolicy.KEEP_LAST,
    depth=1,
)

STATE_SEARCH = "SEARCH"
STATE_COLLECT_DONE = "COLLECT_DONE"
STATE_VISIT_ORDERED = "VISIT_ORDERED"
STATE_LANDING = "LANDING"

# 순회(SEARCH, 필요시 VISIT_ORDERED)가 전부 끝난 뒤에만 진입 — 순회 도중엔 착륙
# 안 함(사용자 요구사항). 착륙 목표는 발견한 마커 중 "그 시점 위치에서 제일 가까운
# 것" 하나를 골라서 감(추가 이동 최소화).
LANDING_PHASE_APPROACH = "APPROACH"   # 목표 마커 좌표 상공까지 위치기반 직선 이동
LANDING_PHASE_VISUAL = "VISUAL"       # 마커가 카메라에 들어온 뒤: ArUco 기준 수평 정렬
LANDING_PHASE_DESCEND = "DESCEND"     # 정렬 유지하며 하강
LANDING_PHASE_FINAL = "FINAL"         # 최종 저속 하강 -> 디스암

# 대회 당일엔 마커 좌표가 랜덤이라 이 개발용 배치 기준 재방문 로직을 지금
# 켜둘 이유가 없음 — 코드는 남겨두되(나중에 필요하면 재활성화) 기본은 off.
ENABLE_VISIT_ORDERED = True

KP_YAW = 1.5
KP_YAW_TURN = 2.0    # 교차점 회전 상태에서 목표 yaw로 수렴시키는 게인
YAW_ALIGN_TOLERANCE = math.radians(5.0)  # 이 안으로 들어오면 회전 완료로 판정
KP_LATERAL = 0.01   # px -> m/s 횡방향 보정
# 무풍 SITL 기준 0.6이었음. windy world(libgazebo_wind_plugin, mean 6m/s/
# gust 18m/s)로 테스트해보니 위치오차가 2.4~3.4m까지 벌어지는 현상 확인 —
# 원인은 PX4가 못 버텨서가 아니라(그 순간 실측 tilt 최대 15.9도, 한계각
# ~45도 대비 여유 충분) _publish_velocity_setpoint()의
# `speed = min(FORWARD_SPEED, dist)`가 오차가 아무리 커도 보정 속도를
# 0.6m/s로 스스로 틀어막고 있었던 것. 2.0으로 올림(20Hz 틱당 이동
# 0.1m로 INTERSECTION_POS_TOLERANCE=0.15보다 작아 오버슈트 위험 없음).
# 주의: 이건 "적응형 바람보정"이 아니라 상한선을 옮긴 것뿐 — 평균풍속
# 9m/s(돌풍 27m/s)로 한 단계만 올려서 재테스트해보니 즉시 실추락(tilt
# 179.8도)함. 즉 이 값의 안전 여유폭은 6~9m/s 사이로 매우 좁음. 진짜
# 피드포워드 보정이 필요하면 PX4 EKF2 wind estimator(/fmu/out/wind)를
# 써야 하는데, 이 기체 파라미터에선 드래그퓨전이 꺼져있어 발행 자체가
# 안 됨(미해결). 상세: [[08 탐색 페이즈 노드 (search_phase_node.py)]]
FORWARD_SPEED = 2.0   # m/s
MAX_YAW_RATE = 0.8    # rad/s
TAKEOFF_ALTITUDE = 2.0    # m, 규정집 "기본 운영고도 약 2m" 기준
TAKEOFF_CLIMB_SPEED = 0.5  # m/s
TAKEOFF_ALT_TOLERANCE = 0.2  # m
ARUCO_CONFIRM_FRAMES = 15
# 바람 등으로 자세가 흔들리면 ArUco 검출/카메라 토픽 프레임이 뜨문뜨문 와서
# ARUCO_CONFIRM_FRAMES 틱이 아예 안 채워질 수 있음 — 그 경우를 위한 시간 기반
# 안전장치(이게 없으면 확인창이 영원히 안 닫혀서 기체가 그 자리에 멈춰버림).
CONFIRM_TIMEOUT_SEC = 4.0
# 확인창이 열려 있는 동안에도 실제 위치가 이 안에 있을 때 검출된 프레임만
# 투표에 반영 — 격자 간격(3.0m)의 절반보다 충분히 작게 잡아서, 옆 칸 마커가
# 잡힌 프레임이 섞여 들어오는 걸 막음(3m/s 바람 실측으로 필요성 확인됨).
ARUCO_VOTE_POS_TOLERANCE = 0.6  # m
# 원래 카메라 HFOV 86°(시뮬레이션 iris_aruco_cam) 가정으로 순회 고도 2m 기준
# 지상 촬영 폭 ~3.7m를 기준 삼아 게이트 1.0m(폭의 절반 대비 53.6%)를 잡았었음.
# 2026-08-27 C270 실물 캘리브레이션(13x8/17mm 체커보드, RMS 0.87px) 결과
# 실측 HFOV는 47.5°(fx=726.6, 640px 기준)로 절반 수준밖에 안 됨 — 2m 고도
# 지상 촬영 폭은 약 1.76m. 같은 비율(53.6%)로 재계산: 0.472m -> 0.47m로 축소.
# marker.pose.position(카메라 프레임)의 평면 오프셋이 이보다 크면 화면 중심에서
# 너무 벗어난 걸로 보고 버림.
ARUCO_LATERAL_GATE = 0.47  # m (2026-08-27 C270 실측 HFOV 47.5° 기준 재계산)
TIMER_PERIOD = 0.05    # 20Hz

# ---- 비전-위치 게이팅 (LineTracker 단독 판정의 오탐/누락을 실측 위치로 보정) ----
# 격자 간격이 3.0m라, 목표 좌표에 이 정도 안 들어온 상태에서 비전이 "교차점"이라고
# 하면 ArUco 마커 경계 등 다른 걸 오검출했을 가능성이 높다고 보고 무시한다.
INTERSECTION_VISION_GATE = 1.5  # m, 이 안에서만 비전 검출을 신뢰
# 반대로 비전이 놓쳐도(그림자·조명 등으로 누락) 실제로 목표 좌표에 이만큼 붙으면
# 무조건 도달로 강제 판정 — 실내 GPS-denied 환경을 흉내내려 시각 검출에만
# 의존하면 오탐/누락 하나로 전체 순회가 어긋나므로, 위치를 최종 백스톱으로 둔다.
# 0.5m였던 걸 0.15로 좁힘 — LineTracker의 intersection 검출이 실측으로 27번 중
# 20번꼴로 누락돼(첫 행 E방향 6칸만 성공, 그 뒤로는 전부 이 백스톱으로 도달 판정)
# 대부분의 회전이 진짜 격자 교차점이 아니라 0.5m(격자 간격 3.0m의 17%) 떨어진
# 지점에서 일어났었음 — 이게 "구획선을 자꾸 놓친다"는 실측 관찰의 원인. 20Hz에
# FORWARD_SPEED=0.6m/s면 틱당 이동거리가 3cm라 0.15m로 좁혀도 오버슈트로 못
# 들어가는 문제는 없음.
INTERSECTION_POS_TOLERANCE = 0.15  # m
MODE_RETRY_PERIOD_SEC = 0.5
MODE_RETRY_EVERY_N_TICKS = max(1, round(MODE_RETRY_PERIOD_SEC / TIMER_PERIOD))

# ---- 최종 착륙 (precision_landing/landing.py의 검증된 게인/상수를 그대로 이식) ----
LANDING_APPROACH_TOLERANCE = 0.5    # m, 이 안에 들어오면 위치기반 접근 끝 -> 시각 서보 전환
# (FORWARD_SPEED=0.6인데 목표속도=min(FORWARD_SPEED,dist)라, 이 문턱을 자연
# 감속 구간(dist<0.6) 안쪽으로 잡아야 전환 시점에 이미 저속임 — 1.0으로 뒀을 땐
# 아직 순항속도라 전환 직후 관성으로 목표를 지나쳐 진동하는 문제가 있었음.)
LANDING_KP_XY = 0.5           # (원래 1.2였음) — ArUco 검출이 뜨문뜨문+EMA 필터
                              # 지연과 겹쳐 이 게인이면 수렴 안 하고 목표 주변을
                              # 계속 도는 한계궤도(limit cycle)가 실측으로 확인됨.
                              # 게인을 낮춰서 지연 대비 과도한 반응을 줄임.
LANDING_MAX_XY_SPEED = 0.5           # m/s (원래 1.2 — 같은 이유로 낮춤)
LANDING_MAX_XY_ACCEL = 1.6           # m/s^2
LANDING_DESCEND_SPEED = 0.1          # m/s
LANDING_CENTER_TOLERANCE = 0.15      # m, 이 안이면 "정렬됨"
LANDING_ALTITUDE = 0.48              # m, 이 고도 밑에서 최종 저속 하강 시작
LANDING_FINAL_SPEED = 0.12           # m/s
LANDING_DISARM_ALTITUDE = 0.05       # m
LANDING_STUCK_TIMEOUT_SEC = 3.0      # 고도 조건은 맞는데 정렬이 안 맞을 때 강행까지 대기시간
LANDING_MARKER_LOST_TIMEOUT = 2.7    # s, 이 시간 넘게 마커 안 보이면 유실 판정
LANDING_MARKER_FILTER_ALPHA = 0.3    # 마커 좌표 EMA 필터 계수


class SearchPhaseNode(Node):

    def __init__(self):
        super().__init__("search_phase_node")

        # 경로점 개수 — 규정상 "대회 당일 통보 -> 비행 전 입력" 값이라 상수가
        # 아니라 ROS 파라미터로 받음. 예: --ros-args -p n_markers_expected:=6
        self.declare_parameter("n_markers_expected", DEFAULT_N_MARKERS_EXPECTED)
        self.n_markers_expected = (
            self.get_parameter("n_markers_expected").get_parameter_value().integer_value
        )

        # 실내 벤치테스트용: GPS/광류 등 위치 소스가 전혀 없으면 PX4 EKF의
        # 로컬 위치추정이 계속 무효(invalid)라서 OFFBOARD ARM이 정상적으로
        # 계속 거부됨(2026-09-21 실측: "위치추정: 불가" 상태에서 ARM 명령이
        # 전부 씹힘). 기본값 False로 두어 실비행 시엔 항상 정상 안전검사를
        # 거치게 하고, 프로펠러를 뗀 지상 벤치테스트에서만 명시적으로
        # --ros-args -p bench_force_arm:=true 로 켜서 force-arm(매직넘버
        # 21196 — 이 파일의 강제 disarm과 동일한 방식)으로 우회한다.
        self.declare_parameter("bench_force_arm", False)
        self.bench_force_arm = (
            self.get_parameter("bench_force_arm").get_parameter_value().bool_value
        )
        if self.bench_force_arm:
            self.get_logger().warn(
                "[BENCH] bench_force_arm=true — 위치추정 무시하고 강제 ARM 시도함. "
                "프로펠러 반드시 제거된 상태에서만 사용할 것.")

        self.bridge = CvBridge()
        self.tracker = LineTracker()

        # ---- 격자/명령 큐 ----
        self.order = zigzag_order(len(X_VALUES), len(Y_VALUES))
        self.command_queue = commands_from_order(self.order)
        self.heading = Heading.E
        self.cur_i, self.cur_j = self.order[0]

        # 실비행/실측 로그로 확인된 사실: 스폰 시 실제 yaw는 East(가정값)가 아니라
        # 대략 North 근방이었고, 이걸 맞춰주는 코드가 없어서 첫 다리(버티포트 ->
        # 첫 격자칸)에서 매번 엉뚱한 방향(대략 90도 어긋난 방향)으로 계속 직진해
        # 맵을 벗어났다. 버티포트=world 원점이므로 첫 칸까지의 방위각을 직접 계산해
        # 이륙 직후 그 방향으로 정확히 정렬시키고 시작한다 — 격자 칸 사이 이동과
        # 달리 이 첫 구간은 대각선이라 4방위(N/E/S/W)로 딱 떨어지지 않는다.
        first_x, first_y = grid_xy(*self.order[0])
        self.turning = True        # True면 목표 yaw로 회전 완료할 때까지 전진 정지
        self.target_yaw = math.atan2(first_y, first_x)

        # ---- 상태 머신 ----
        self.state = STATE_SEARCH
        self.found_markers = {}   # marker_id -> {"grid": (i,j), "world_pose": (x,y)}
        self._aruco_vote_buffer = []
        self._at_intersection_cooldown = 0  # 같은 교차점에서 중복 트리거 방지
        self._confirming = False  # ArUco 확인창이 열려 있는 동안 True — 이동/재트리거 정지

        # ---- 최종 착륙 상태 (순회 다 끝난 뒤에만 씀) ----
        self.landing_phase = None
        self.landing_target_id = None
        self.landing_target_xy = None
        self.land_vx = self.land_vy = self.land_vz = 0.0
        self.marker_cam_x = self.marker_cam_y = self.marker_cam_z = 0.0
        self.marker_visible = False
        self.marker_ever_seen_landing = False
        self.last_marker_time = None
        self._land_stuck_since = None
        self.land_disarm_sent = False

        # ---- PX4 상태 ----
        self.arming_state = None
        self.nav_state = None
        self.drone_x = self.drone_y = self.drone_z = 0.0
        self.current_yaw = 0.0  # ENU 기준(0=East, 반시계+), on_local_position에서 갱신
        self.local_position_received = False
        self.offboard_setpoint_counter = 0
        self._mode_retry_tick = 0  # OFFBOARD 전환+ARM 재시도 주기용 (landing.py와 동일 패턴)

        # ---- Subscribers ----
        self.create_subscription(Image, "/camera/image_raw", self.on_image, IMAGE_QOS)
        self.create_subscription(ArucoDetection, "/aruco_detections", self.on_aruco, 10)
        self.create_subscription(
            VehicleStatus, "/fmu/out/vehicle_status_v1", self.on_status, PX4_QOS)
        self.create_subscription(
            VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1",
            self.on_local_position, PX4_QOS)

        # ---- Publishers (PX4) ----
        self.offboard_pub = self.create_publisher(
            OffboardControlMode, "/fmu/in/offboard_control_mode", PX4_QOS)
        self.traj_pub = self.create_publisher(
            TrajectorySetpoint, "/fmu/in/trajectory_setpoint", PX4_QOS)
        self.cmd_pub = self.create_publisher(
            VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)

        # ---- Publisher (RViz) ----
        self.viz_pub = self.create_publisher(MarkerArray, "/kasa_search_markers", 10)

        self._last_track = LineTrackResult(False, 0.0, 0.0, False)
        self.timer = self.create_timer(TIMER_PERIOD, self.control_tick)

        self.get_logger().info(
            f"탐색 페이즈 시작 — 방문 순서 {len(self.order)}칸, "
            f"목표 마커 {self.n_markers_expected}개"
        )

    # ------------------------------------------------------------------
    # PX4 콜백
    # ------------------------------------------------------------------

    def on_status(self, msg):
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    def on_local_position(self, msg):
        # PX4 local position은 NED(x=North,y=East,z=Down). world는 ENU라
        # x<->y가 뒤바뀌고 z 부호가 반대다 — 여기선 world(x=East,y=North)로 맞춰줌.
        self.drone_x = msg.y
        self.drone_y = msg.x
        self.drone_z = -msg.z

        # NED heading(0=North, 시계방향 +) -> ENU 기준 yaw(0=East, 반시계 +)로 변환.
        # 이전엔 이 실제 yaw 대신 격자용 이산 방위(self.heading, N/E/S/W)로
        # body->world 회전을 계산해서, yaw_rate로 실제 기수를 계속 틀면서도
        # 속도벡터 회전 기준은 안 바뀌는 불일치가 생겼었음(불안정 비행의 원인).
        heading = getattr(msg, "heading", None)
        if heading is not None:
            self.current_yaw = (math.pi / 2 - heading + math.pi) % (2 * math.pi) - math.pi
        self.local_position_received = True

    # ------------------------------------------------------------------
    # 비전 콜백
    # ------------------------------------------------------------------

    def on_image(self, msg: Image):
        # 회전 중엔 전진/라인추종 자체를 안 하니 비전 처리도 건너뜀 — 회전이
        # 끝나야 새 방향 기준 라인 판독이 의미가 있음.
        if self.turning:
            return

        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        self._last_track = self.tracker.process(cv_image)

        if self._at_intersection_cooldown > 0:
            self._at_intersection_cooldown -= 1
            return

        # 비전 검출은 목표 좌표 근처에서만 신뢰한다 — ArUco 마커 경계 등
        # 격자 교차점이 아닌 걸 오검출해도 위치상 말이 안 되면 무시.
        if (not self._confirming and self._last_track.intersection
                and self._distance_to_target() < INTERSECTION_VISION_GATE):
            self._on_intersection_reached()
            self._at_intersection_cooldown = 10  # 한 교차점에서 여러 번 안 튀게

    def _distance_to_target(self):
        if not self.local_position_received:
            return float("inf")
        tx, ty = grid_xy(self.cur_i, self.cur_j)
        return math.hypot(self.drone_x - tx, self.drone_y - ty)

    def _distance_to_confirm_cell(self):
        """cur_i/cur_j와 별개로 _confirm_cell 기준 거리 — 개념상 확인창 동안엔
        cur_i/cur_j가 그대로라 _distance_to_target()과 같은 값이지만, 무엇을
        재는지 헷갈리지 않게 확인 로직 쪽은 이걸로 명시적으로 뺌."""
        if not self.local_position_received:
            return float("inf")
        tx, ty = grid_xy(*self._confirm_cell)
        return math.hypot(self.drone_x - tx, self.drone_y - ty)

    def _check_position_fallback(self):
        """비전이 교차점을 놓쳐도(그림자·조명 등) 실제로 목표 좌표에 충분히
        가까워지면 도달로 강제 판정 — 시각 검출 하나에만 기대면 오탐/누락
        한 번으로 전체 순회 인덱스가 실제 위치와 어긋나버리는 걸 막는 안전망."""
        if (self.turning or self._at_intersection_cooldown > 0 or self._confirming
                or not self.command_queue):
            return
        if self._distance_to_target() < INTERSECTION_POS_TOLERANCE:
            self.get_logger().info(
                f"[게이팅] 비전 미검출 -> 위치 기준 강제 도달 판정 "
                f"({self.cur_i},{self.cur_j})"
            )
            self._on_intersection_reached()
            self._at_intersection_cooldown = 10

    def _apply_command(self, cmd):
        """커맨드 큐에서 꺼낸 명령을 실제로 반영. 방향 전환이 필요한 명령이면
        전진을 멈추고 목표 yaw로 회전을 완료할 때까지 self.turning을 켠다 —
        예전엔 self.heading 라벨만 바뀌고 실제 회전을 시키는 코드가 없어서,
        상태머신은 다음 칸으로 넘어갔다고 믿는데 기체는 원래 방향 그대로 직진하는
        불일치가 있었음(실비행 로그로 확인됨: yaw가 실제로는 안 바뀜)."""
        self.heading = cmd.heading
        self.cur_i, self.cur_j = cmd.target_i, cmd.target_j
        if cmd.turn != TurnCmd.STRAIGHT:
            self.turning = True
            self.target_yaw = _HEADING_YAW[cmd.heading]

    def _on_intersection_reached(self):
        # 다음 칸으로 넘어가는 건 ArUco 확인창이 완전히 닫힌 뒤로 미룬다
        # (_advance_after_intersection에서 함) — 예전엔 여기서 바로 다음 명령을
        # 적용해서 확인창이 열려 있는 동안에도 기체가 계속 다음 목표로 이동을
        # 시작했는데, 바람으로 그 몇 프레임 사이 이동거리가 커지면 지금 칸이
        # 아니라 다음 칸의 마커를 봐놓고 지금 칸(_confirm_cell) 걸로 기록해버리는
        # 오기록이 실측(SITL, 3m/s 바람)으로 확인됨. cur_i/cur_j를 그대로 두면
        # 속도제어가 자연히 지금 칸 위에서 계속 맴돌아서(추가 상태 없이) 확인이
        # 끝날 때까지 실질적으로 정지해 있는 효과를 냄.
        self._begin_aruco_confirmation()

    def _advance_after_intersection(self):
        """ArUco 확인창이 닫힌 뒤(_finalize_aruco_vote 끝에서) 호출 — 다음 칸
        명령 적용 + 상태 전환 체크. 예전 _on_intersection_reached()의 나머지 절반."""
        if self.state == STATE_SEARCH:
            if self.command_queue:
                cmd = self.command_queue.popleft()
                self._apply_command(cmd)
                self.get_logger().info(
                    f"[SEARCH] 교차점 도달 ({self.cur_i},{self.cur_j}) "
                    f"-> {cmd.turn.value} -> heading={cmd.heading.name}"
                )
            # 28칸 전체 순회 자체는 실측으로 검증 완료(2026-08-24). 이제부터는
            # 규정집(중급) ⑥번대로: N개 다 찾으면 지그재그 중단하고 구조
            # 경로(VISIT_ORDERED)로 넘어간다.
            if ENABLE_VISIT_ORDERED and len(self.found_markers) >= self.n_markers_expected:
                self._transition_to_visit_ordered()

        elif self.state == STATE_VISIT_ORDERED:
            if self.command_queue:
                cmd = self.command_queue.popleft()
                self._apply_command(cmd)
                self.get_logger().info(
                    f"[VISIT_ORDERED] {cmd.turn.value} -> heading={cmd.heading.name}"
                )
            else:
                self.get_logger().info("[VISIT_ORDERED] 마지막 경로점 도달 — 착륙 단계로 넘김")

    def _begin_aruco_confirmation(self):
        self._aruco_vote_buffer = []
        self._confirm_deadline_ticks = ARUCO_CONFIRM_FRAMES
        self._confirming = True
        self._confirm_deadline_time = self.get_clock().now() + Duration(seconds=CONFIRM_TIMEOUT_SEC)
        # 지금(교차점에 방금 도착한) 칸을 미리 찍어둔다 — cur_i/cur_j는
        # _advance_after_intersection(확인창 닫힌 뒤)까지 안 바뀌니 지금은
        # 그냥 self.cur_i/cur_j를 써도 되지만, 과거에 "다음 칸으로 먼저
        # 넘어간 뒤 뒤늦게 확정"하던 구조에서 한 칸 밀려 기록되는 버그가
        # 있었던 자리라 안전하게 그대로 스냅샷 떠서 씀.
        self._confirm_cell = (self.cur_i, self.cur_j)

    def on_aruco(self, msg: ArucoDetection):
        if self._confirm_window_open():
            # cur_i/cur_j를 확인창 동안 얼려놔도, 바람에 순간적으로 다음 칸
            # 쪽으로 밀린 틈에 다음 칸 마커가 카메라에 잡히면 그 프레임도
            # 그냥 투표에 들어가버려서 확정 위치가 밀리는 문제가 실측(SITL,
            # 3m/s 바람)으로 남아있었음 — 그 순간 실제 위치가 이 칸(confirm_cell)
            # 근처가 아니면 그 프레임의 검출은 아예 투표에서 뺀다.
            if self._distance_to_confirm_cell() < ARUCO_VOTE_POS_TOLERANCE:
                # 드론 위치가 맞아도 자세가 조금만 기울어도(바람 버티느라 상시
                # 기울어 있음) 옆 칸 마커까지 화면에 들어올 수 있음(시뮬레이션
                # 카메라 HFOV 86° 기준 실측으로 확인됐던 문제). C270 실카메라는
                # HFOV가 47.5°로 더 좁아서(2m 고도 지상 촬영 폭 ~1.76m) 이 문제
                # 자체는 완화되지만, 게이트는 그대로 유지 — 마커 자체의 카메라
                # 상대좌표(화면 중심에서 얼마나 벗어났는지)로 한 번 더 걸러서,
                # 화면 가장자리에 걸린(=옆 칸 마커일 가능성이 높은) 검출은
                # 투표에서 뺀다.
                for marker in msg.markers:
                    lateral = math.hypot(
                        marker.pose.position.x, marker.pose.position.y)
                    if lateral < ARUCO_LATERAL_GATE:
                        self._aruco_vote_buffer.append(marker.marker_id)
            self._confirm_deadline_ticks -= 1
            if self._confirm_deadline_ticks <= 0:
                self._finalize_aruco_vote()

        if self.state == STATE_LANDING:
            self._on_aruco_landing(msg)

    def _on_aruco_landing(self, msg: ArucoDetection):
        """착륙 목표 마커(landing_target_id)만 골라 카메라 좌표를 EMA
        필터링 — precision_landing/landing.py의 aruco_callback과 동일 로직."""
        target = next(
            (m for m in msg.markers if m.marker_id == self.landing_target_id), None)
        if target is None:
            return
        raw_x = target.pose.position.x
        raw_y = target.pose.position.y
        raw_z = target.pose.position.z
        a = LANDING_MARKER_FILTER_ALPHA
        if self.marker_ever_seen_landing:
            self.marker_cam_x = a * raw_x + (1 - a) * self.marker_cam_x
            self.marker_cam_y = a * raw_y + (1 - a) * self.marker_cam_y
            self.marker_cam_z = a * raw_z + (1 - a) * self.marker_cam_z
        else:
            self.marker_cam_x, self.marker_cam_y, self.marker_cam_z = raw_x, raw_y, raw_z
        self.marker_visible = True
        self.marker_ever_seen_landing = True
        self.last_marker_time = self.get_clock().now()

    def _confirm_window_open(self):
        return self._confirming

    def _check_confirm_timeout(self):
        """매 control_tick에서 확인창이 CONFIRM_TIMEOUT_SEC 넘게 안 닫히면
        강제로 닫는다 — aruco_tracker가 그 사이 토픽을 아예 안 쏘면(글리치)
        on_aruco()의 틱 카운트다운 자체가 안 돌아서 _confirm_deadline_ticks만
        으론 절대 안 끝나고 기체가 그 자리에 영원히 멈추는 걸 막는 안전장치."""
        if not self._confirming:
            return
        if self.get_clock().now() >= self._confirm_deadline_time:
            self.get_logger().warn("[게이팅] ArUco 확인 타임아웃 -> 강제 진행")
            self._finalize_aruco_vote()

    def _finalize_aruco_vote(self):
        self._confirming = False
        if self._aruco_vote_buffer:
            marker_id = statistics.mode(self._aruco_vote_buffer)
            if marker_id != VERTIPORT_MARKER_ID and marker_id not in self.found_markers:
                # 버티포트 중앙 마커(ID0)는 미션 경로점이 아님 — 카운트 안 함
                cell = self._confirm_cell
                self.found_markers[marker_id] = {
                    "grid": cell,
                    "world_pose": grid_xy(*cell),
                }
                self.get_logger().info(
                    f"ArUco ID{marker_id} 확정 @ grid{cell} "
                    f"world{grid_xy(*cell)} "
                    f"— 총 {len(self.found_markers)}/{self.n_markers_expected}"
                )
        self._advance_after_intersection()

    # ------------------------------------------------------------------
    # 상태 전환
    # ------------------------------------------------------------------

    def _transition_to_visit_ordered(self):
        self.state = STATE_COLLECT_DONE
        # 규정(중급) ⑥번: "최적 경로는 ArUcoMarker 식별번호 순서로 결정(예:
        # 1→2→3→4), 최적 구조 경로(역순)에 따라 이동" — 그래서 방문 순서는
        # 내림차순(마지막에 찾은 마커부터 먼저 복귀 방향으로).
        ordered_ids = sorted(self.found_markers, reverse=True)
        self.get_logger().info(
            f"마커 전부 확정 — 지그재그 중단. 구조 경로(역순): {ordered_ids}"
        )
        order_cells = [self.found_markers[mid]["grid"] for mid in ordered_ids]
        # 지금 있는 칸부터 역순 ID 순서로 다시 지그재그 큐를 짬 —
        # 실제 이동은 여전히 라인트레이싱(격자선을 따라가는 것)이라 가정하고,
        # 방문할 칸 사이의 최단 격자 경로는 각 축을 순서대로 맞추는 방식으로 생성.
        route = [(self.cur_i, self.cur_j)] + order_cells
        commands = collections.deque()
        heading = self.heading
        for (i0, j0), (i1, j1) in zip(route, route[1:]):
            path = expand_axis_path((i0, j0), (i1, j1))
            commands.extend(commands_from_order(path, start_heading=heading))
            heading = commands[-1].heading if commands else heading
        self.command_queue = commands
        self.state = STATE_VISIT_ORDERED

    def _check_mission_complete(self):
        """순회(SEARCH, 필요시 VISIT_ORDERED)가 완전히 끝났는지 매 틱 확인.
        비전 재트리거처럼 안정성이 떨어지는 이벤트에 기대지 않고, "칐 큐가
        비었고 + 실제로 마지막 목표 좌표에 도착했다"를 직접 확인해서 착륙으로
        전환한다 — 순회 도중엔 착륙하면 안 된다는 요구사항 그대로, 큐가 남아있는
        동안은 절대 여기서 전환 안 됨."""
        if self.state not in (STATE_SEARCH, STATE_VISIT_ORDERED):
            return
        if self.command_queue or self.turning or self._confirming:
            return
        if self._distance_to_target() < INTERSECTION_POS_TOLERANCE:
            self._transition_to_landing()

    def _transition_to_landing(self):
        # 규정(중급) ⑦번: 자동 착륙은 찾은 마커 위가 아니라 버티포트(시작지점)
        # 복귀 후 1회. 버티포트 = world 원점(0,0)이고, 정밀착륙 시각서보는
        # 버티포트 중앙의 VERTIPORT_MARKER_ID(ID0) 마커를 기준으로 한다.
        self.landing_target_id = VERTIPORT_MARKER_ID
        self.landing_target_xy = (0.0, 0.0)
        self.landing_phase = LANDING_PHASE_APPROACH
        self.state = STATE_LANDING
        self.get_logger().info(
            "[LANDING] 순회 완료 — 버티포트(0,0)로 복귀 후 착륙 시작"
        )

    # ------------------------------------------------------------------
    # 제어 루프
    # ------------------------------------------------------------------

    def control_tick(self):
        self._publish_offboard_heartbeat()
        self._check_position_fallback()
        self._check_confirm_timeout()
        self._check_mission_complete()
        self._publish_velocity_setpoint()
        self._publish_viz()

        # 착륙 완료(강제 disarm까지 보낸) 뒤에는 재시도를 멈춘다 — 안 멈추면
        # ensure_offboard_and_armed()가 "안 armed"를 보고 곧바로 다시 ARM
        # 명령을 보내버려서, 착륙 직후 자동으로 재무장되고 다시 못 뜨는 상태로
        # 계속 갇히는 문제가 있었음(실측으로 확인 — "착륙은 하는데 다시
        # 이륙을 못한다"). 규정 ⑦번(자동 착륙)은 종료 상태라 재무장이 아니라
        # 여기서 끝나야 정상.
        if self.land_disarm_sent:
            return

        # PX4는 OFFBOARD 진입 요청 전에 setpoint 스트림이 먼저 몇 틱 이상
        # 흐르고 있어야 받아준다 (landing.py에서 검증된 순서 그대로 따름).
        if self.offboard_setpoint_counter < 10:
            self.offboard_setpoint_counter += 1
            return
        self.ensure_offboard_and_armed()

    def _publish_offboard_heartbeat(self):
        msg = OffboardControlMode()
        msg.position = False
        msg.velocity = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.offboard_pub.publish(msg)

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def ensure_offboard_and_armed(self):
        """PX4를 OFFBOARD 모드로 전환 + ARM. 확인될 때까지 주기적으로 재시도
        (landing.py의 ensure_offboard_and_armed와 동일한 패턴 — 명령 1회 전송이
        타이밍상 씹힐 수 있어서 재시도가 필요함)."""
        offboard = (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        armed = (self.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        if offboard and armed:
            return

        self._mode_retry_tick += 1
        if self._mode_retry_tick < MODE_RETRY_EVERY_N_TICKS:
            return
        self._mode_retry_tick = 0

        if not offboard:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        if not armed:
            if self.bench_force_arm:
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=1.0, param2=21196.0)
            else:
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)

    def _publish_velocity_setpoint(self):
        # LANDING은 이륙 고도유지 체크보다 먼저 걸러야 함 — 아래 이륙 climb 체크
        # (drone_z < TAKEOFF_ALTITUDE-TAKEOFF_ALT_TOLERANCE = 1.8m)가 LANDING보다
        # 먼저 있으면, 착륙 하강으로 고도가 1.8m 밑으로 내려가는 순간 이 체크가
        # LANDING을 무시하고 강제로 다시 위로 띄워버려서 1.8m 근처에서 하강이
        # 절대 못 뚫고 계속 튕기는 버그가 있었음(실측으로 확인 — 사용자가 "이륙
        # 고도유지 로직과 충돌하는 것 같다"고 정확히 지적함).
        if self.state == STATE_LANDING:
            self._publish_landing_setpoint()
            return

        # PX4 착륙감지기(vehicle_land_detected)는 순수 수평 속도만으론 안 풀린다 —
        # armed+offboard여도 "landed: True"인 채로 수평 setpoint를 무시함
        # (실비행 테스트로 직접 확인됨). 규정 ①번(자동 수직 이륙) 그대로,
        # 목표 고도(TAKEOFF_ALTITUDE)까지 순수 상승만 먼저 명령해서 이 상태부터 벗어남.
        if self.drone_z < TAKEOFF_ALTITUDE - TAKEOFF_ALT_TOLERANCE:
            msg = TrajectorySetpoint()
            msg.position = [float("nan")] * 3
            msg.velocity = [0.0, 0.0, -TAKEOFF_CLIMB_SPEED]  # NED z: 상승은 음수
            msg.yaw = float("nan")  # yaw=0.0(기본값)을 두면 PX4가 그걸 절대 yaw
            # 목표로 계속 붙잡아서 yawspeed 명령과 계속 싸운다 — NaN이어야
            # "yaw는 안 건드림"으로 해석되고 yawspeed만 단독으로 먹힌다
            # (원인 확인: 12시 순회 테스트 중 turning 상태에서 yawspeed를 거의
            # 최대치로 12초 넘게 명령해도 실측 yaw가 1도도 안 움직이는 걸로 확정).
            msg.yawspeed = 0.0
            msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
            self.traj_pub.publish(msg)
            return

        if self.turning:
            self._publish_turn_setpoint()
            return

        # 진행 방향은 항상 목표 좌표(known ground truth, EKF 실측 위치 기준)로
        # 직접 유도한다 — LineTracker의 e_y/e_psi는 노이즈가 심해서(실측: 프레임당
        # vy 7m/s 점프, yawspeed 91deg/s 점프) 이걸로 직접 조향하면 목표를 훌쩍
        # 지나쳐버리는 게 실비행으로 확인됨. 대신 라인트레이싱(비전)은 "교차점
        # 도달 판정"(on_image의 INTERSECTION_VISION_GATE)에만 쓴다 — 위치는
        # 항상 맞는 곳으로 가게 하고, 비전은 언제 도착 판정할지를 더 빠르게/
        # 정확하게 확인해주는 역할로 분리한 것. 이게 이 프로젝트의 이산-연속
        # 융합 측위 아이디어를 SITL(정확한 EKF 위치 사용 가능)에 맞게 단순화한 버전.
        yaw_rate = 0.0
        tx, ty = grid_xy(self.cur_i, self.cur_j)
        dx, dy = tx - self.drone_x, ty - self.drone_y
        dist = math.hypot(dx, dy)
        if dist > 1e-3:
            speed = min(FORWARD_SPEED, dist)
            vx_world, vy_world = dx / dist * speed, dy / dist * speed
        else:
            vx_world = vy_world = 0.0

        msg = TrajectorySetpoint()
        msg.position = [float("nan")] * 3
        # world(ENU: x=East,y=North) -> PX4 local(NED: x=North,y=East,z=Down)
        msg.velocity = [vy_world, vx_world, 0.0]
        msg.yaw = float("nan")  # yaw=NaN이어야 yawspeed 단독 제어가 실제로 먹힘
        msg.yawspeed = yaw_rate
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(msg)

    def _publish_turn_setpoint(self):
        """교차점 회전 상태: 전진/횡이동 없이 목표 yaw(self.target_yaw)로만 제자리
        회전. 오차가 YAW_ALIGN_TOLERANCE 안으로 들어오면 회전 완료로 보고
        self.turning을 끈다 — 그 다음 tick부터 라인트레이싱이 재개된다.

        yawspeed 부호: current_yaw는 NED heading을 ENU(E=0, 반시계+)로 변환한
        값이라 heading_rate = -current_yaw_rate 관계다. TrajectorySetpoint.yawspeed는
        NED 기준(양수=시계방향)이므로, current_yaw를 늘리려면(반시계 회전)
        yawspeed는 음수여야 한다 — 그래서 부호가 -KP_YAW_TURN*yaw_err.
        """
        yaw_err = (self.target_yaw - self.current_yaw + math.pi) % (2 * math.pi) - math.pi

        msg = TrajectorySetpoint()
        msg.position = [float("nan")] * 3
        msg.velocity = [0.0, 0.0, 0.0]
        msg.yaw = float("nan")
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)

        if abs(yaw_err) < YAW_ALIGN_TOLERANCE:
            self.turning = False
            msg.yawspeed = 0.0
        else:
            msg.yawspeed = max(
                -MAX_YAW_RATE, min(MAX_YAW_RATE, -KP_YAW_TURN * yaw_err))
        self.traj_pub.publish(msg)

    # ------------------------------------------------------------------
    # 최종 착륙 — precision_landing/landing.py의 검증된 상태머신 이식
    # ------------------------------------------------------------------

    def _publish_landing_setpoint(self):
        dt = TIMER_PERIOD

        if self.landing_phase == LANDING_PHASE_APPROACH:
            tx, ty = self.landing_target_xy
            dx, dy = tx - self.drone_x, ty - self.drone_y
            dist = math.hypot(dx, dy)
            if dist < LANDING_APPROACH_TOLERANCE:
                self.get_logger().info("[LANDING] 목표 마커 근접 — 시각 서보 정렬 시작")
                self.landing_phase = LANDING_PHASE_VISUAL
                # self.land_vx/vy를 안 끊고 그대로 아래 VISUAL 램핑으로 넘김 —
                # 여기서 속도를 갑자기 0으로 끊거나 unramped 값을 쓰면, 실제
                # 기체는 관성으로 계속 움직이는데 목표 속도만 뚝 떨어져서 지나쳤다
                # 되돌아오는 진동이 생김(실비행으로 확인됨 — 착지 안 되고 목표
                # 주변을 왔다갔다 계속 돔). 그래서 이 구간도 아래와 동일하게
                # self.land_vx/vy를 ramp_toward로만 갱신하고 즉시 반영 안 함.
            else:
                # 거리가 멀 땐 FORWARD_SPEED로, 가까워질수록(특히 전환 문턱
                # LANDING_APPROACH_TOLERANCE 부근) 자연히 감속되도록 dist 자체를
                # 상한으로 같이 씀 — 전환 시점에 이미 저속이어야 위 문제가 안 생김.
                speed = min(FORWARD_SPEED, dist)
                target_vx, target_vy = dx / dist * speed, dy / dist * speed
                self.land_vx = self._ramp_toward(self.land_vx, target_vx, dt, LANDING_MAX_XY_ACCEL)
                self.land_vy = self._ramp_toward(self.land_vy, target_vy, dt, LANDING_MAX_XY_ACCEL)
                self.land_vz = self._ramp_toward(self.land_vz, 0.0, dt, LANDING_MAX_XY_ACCEL)
                self._send_landing_velocity(self.land_vx, self.land_vy, self.land_vz)
                return
        stale = self._landing_marker_stale()
        altitude_ok = self.drone_z < LANDING_ALTITUDE
        err = math.hypot(self.marker_cam_x, self.marker_cam_y)

        if stale and not (altitude_ok and
                           self.landing_phase in (LANDING_PHASE_VISUAL, LANDING_PHASE_DESCEND,
                                                   LANDING_PHASE_FINAL)):
            # 마커를 못 보고 있을 때 그냥 제자리 정지시키면, 실측(SITL)으로 확인됨:
            # 위치기반 접근이 끝난 지점이 카메라 인식엔 충분히 안 가까울 수 있고
            # (실측 오차 1m 이상), 정지한 채로는 그 오차를 절대 못 줄여서 마커를
            # 영영 재획득 못 함. 그래서 완전히 손 놓지 않고 알고 있는 목표 좌표
            # (landing_target_xy)로 계속 접근시킨다 — 탐색 단계에서 쓴 것과 같은
            # "비전 공백은 위치로 메운다" 방식.
            tx, ty = self.landing_target_xy
            dx, dy = tx - self.drone_x, ty - self.drone_y
            dist = math.hypot(dx, dy)
            if dist > 1e-3:
                speed = min(FORWARD_SPEED, dist)
                target_vx, target_vy = dx / dist * speed, dy / dist * speed
            else:
                target_vx = target_vy = 0.0
            self.land_vx = self._ramp_toward(self.land_vx, target_vx, dt, LANDING_MAX_XY_ACCEL)
            self.land_vy = self._ramp_toward(self.land_vy, target_vy, dt, LANDING_MAX_XY_ACCEL)
            self.land_vz = self._ramp_toward(self.land_vz, 0.0, dt, LANDING_MAX_XY_ACCEL)
            self._send_landing_velocity(self.land_vx, self.land_vy, self.land_vz)
            return

        if altitude_ok:
            if self._land_stuck_since is None:
                self._land_stuck_since = self.get_clock().now()
            stuck_elapsed = (
                self.get_clock().now() - self._land_stuck_since).nanoseconds / 1e9
        else:
            self._land_stuck_since = None
            stuck_elapsed = 0.0

        if self.landing_phase in (LANDING_PHASE_VISUAL, LANDING_PHASE_DESCEND) and altitude_ok:
            aligned = err < LANDING_CENTER_TOLERANCE
            timed_out = stuck_elapsed > LANDING_STUCK_TIMEOUT_SEC
            if aligned or timed_out or stale:
                reason = "정렬됨" if aligned else ("정체 -> 강행" if timed_out else "마커 유실 -> 저고도 강행")
                self.get_logger().info(
                    f"[LANDING] 고도 {LANDING_ALTITUDE:.2f}m 미만 ({reason}) -> 최종 착륙")
                self.landing_phase = LANDING_PHASE_FINAL

        if self.landing_phase == LANDING_PHASE_VISUAL:
            target_vx, target_vy = self._landing_position_command(err)
            self.land_vx = self._ramp_toward(self.land_vx, target_vx, dt, LANDING_MAX_XY_ACCEL)
            self.land_vy = self._ramp_toward(self.land_vy, target_vy, dt, LANDING_MAX_XY_ACCEL)
            self.land_vz = self._ramp_toward(self.land_vz, 0.0, dt, LANDING_MAX_XY_ACCEL)
            if err < LANDING_CENTER_TOLERANCE:
                self.landing_phase = LANDING_PHASE_DESCEND

        elif self.landing_phase == LANDING_PHASE_DESCEND:
            target_vx, target_vy = self._landing_position_command(err)
            self.land_vx = self._ramp_toward(self.land_vx, target_vx, dt, LANDING_MAX_XY_ACCEL)
            self.land_vy = self._ramp_toward(self.land_vy, target_vy, dt, LANDING_MAX_XY_ACCEL)
            self.land_vz = LANDING_DESCEND_SPEED  # NED: 아래로 이동 = 양수
            if err > LANDING_CENTER_TOLERANCE * 2:
                self.landing_phase = LANDING_PHASE_VISUAL

        elif self.landing_phase == LANDING_PHASE_FINAL:
            # NAV_LAND로 PX4 자체 AUTO.LAND에 안 넘김 — 넘기면 MPC_LAND_SPEED(보통
            # 더 빠름)로 바뀌어 갑자기 빨리 착륙하는 문제가 있었음(landing.py에서
            # 이미 겪은 문제, 그대로 적용). 끝까지 직접 저속 하강 후 여기서 disarm.
            self.land_vx = self._ramp_toward(self.land_vx, 0.0, dt, LANDING_MAX_XY_ACCEL)
            self.land_vy = self._ramp_toward(self.land_vy, 0.0, dt, LANDING_MAX_XY_ACCEL)
            self.land_vz = self._ramp_toward(self.land_vz, LANDING_FINAL_SPEED, dt, LANDING_MAX_XY_ACCEL)

            if not self.land_disarm_sent and self.drone_z <= LANDING_DISARM_ALTITUDE:
                self.land_vx = self.land_vy = self.land_vz = 0.0
                # param2=21196: PX4의 "force disarm" 매직넘버. vehicle_land_detected가
                # (OFFBOARD 속도제어 중이라) "landed"로 확정 안 된 상태에서는 일반
                # disarm 명령을 안전장치로 그냥 무시해버림 — 실측으로 확인됨(고도
                # 0.05m 이하에서 disarm 로그는 찍히는데 arming_state가 계속 ARMED로
                # 안 바뀜). 이미 착지 직전 최종 단계라 강제로 밀어붙여도 안전함.
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0, param2=21196.0)
                self.get_logger().info(
                    f"[LANDING] 고도 {LANDING_DISARM_ALTITUDE:.2f}m 이하 -> 강제 disarm")
                self.land_disarm_sent = True
                self._log_mission_summary()

        self._send_landing_velocity(self.land_vx, self.land_vy, self.land_vz)

    def _landing_position_command(self, err):
        """마커의 카메라 좌표(marker_cam_x/y)를 world 속도로.

        landing.py 원본은 body_y = marker_cam_x(부호 그대로)였는데, 실비행
        2회(서로 다른 yaw ~0°/~180°)에서 공통으로 World_X축(=body_x=-marker_cam_y
        가 담당)은 잘 수렴하고 World_Y축(=body_y가 담당)만 5m대 진폭으로 계속
        진동하는 게 재현됨 — yaw가 달라도 실패하는 축이 항상 body_y 쪽이라 회전
        공식이 아니라 이 부호 자체가 원인으로 판단, 뒤집어서 수정."""
        if err < 1e-6:
            return 0.0, 0.0
        body_x = -self.marker_cam_y
        body_y = -self.marker_cam_x
        speed = min(LANDING_MAX_XY_SPEED, LANDING_KP_XY * err)
        ux, uy = body_x / err, body_y / err
        vx_body, vy_body = ux * speed, uy * speed
        ch, sh = math.cos(self.current_yaw), math.sin(self.current_yaw)
        vx_world = vx_body * ch - vy_body * sh
        vy_world = vx_body * sh + vy_body * ch
        return vx_world, vy_world

    def _landing_marker_stale(self):
        if not self.marker_ever_seen_landing:
            return True
        elapsed = (self.get_clock().now() - self.last_marker_time).nanoseconds / 1e9
        return elapsed > LANDING_MARKER_LOST_TIMEOUT

    def _log_mission_summary(self):
        """착륙(강제 disarm) 직후 1회 — 확정된 조난자(마커) 위치를 ID 순으로
        한 줄씩 정리해서 로그로 남김 (GUI 로그창에 그대로 찍힘, '로그 저장'으로
        파일로도 남길 수 있음)."""
        for mid in sorted(self.found_markers):
            gx, gy = self.found_markers[mid]["grid"]
            wx, wy = self.found_markers[mid]["world_pose"]
            self.get_logger().info(
                f"[SUMMARY] ID{mid}: grid({gx},{gy}) world({wx:.2f}, {wy:.2f})"
            )

    def _send_landing_velocity(self, vx_world, vy_world, vz_ned):
        msg = TrajectorySetpoint()
        msg.position = [float("nan")] * 3
        msg.velocity = [vy_world, vx_world, vz_ned]
        msg.yaw = float("nan")
        msg.yawspeed = 0.0
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.traj_pub.publish(msg)

    @staticmethod
    def _ramp_toward(current, target, dt, max_accel):
        max_delta = max_accel * dt
        delta = target - current
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        return current + delta

    # ------------------------------------------------------------------
    # RViz 시각화 — grid_marker.py가 그리는 정적 격자 위에 탐색 진행 상황을 덧그림
    # ------------------------------------------------------------------

    def _publish_viz(self):
        arr = MarkerArray()

        cur = self._header(Marker())
        cur.ns, cur.id = "search_current_cell", 0
        cur.type, cur.action = Marker.SPHERE, Marker.ADD
        cx, cy = grid_xy(self.cur_i, self.cur_j)
        cur.pose.position.x, cur.pose.position.y, cur.pose.position.z = cx, cy, 0.3
        cur.pose.orientation.w = 1.0
        cur.scale.x = cur.scale.y = cur.scale.z = 0.6
        cur.color.r, cur.color.g, cur.color.b, cur.color.a = 0.0, 0.6, 1.0, 1.0
        arr.markers.append(cur)

        for idx, mid in enumerate(sorted(self.found_markers)):
            info = self.found_markers[mid]
            mx, my = info["world_pose"]

            dot = self._header(Marker())
            dot.ns, dot.id = "search_found_markers", mid
            dot.type, dot.action = Marker.CYLINDER, Marker.ADD
            dot.pose.position.x, dot.pose.position.y, dot.pose.position.z = mx, my, 0.05
            dot.pose.orientation.w = 1.0
            dot.scale.x = dot.scale.y = 0.5
            dot.scale.z = 0.1
            dot.color.r, dot.color.g, dot.color.b, dot.color.a = 0.0, 1.0, 0.2, 0.9
            arr.markers.append(dot)

            label = self._header(Marker())
            label.ns, label.id = "search_found_labels", mid
            label.type, label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
            label.pose.position.x, label.pose.position.y = mx, my
            label.pose.position.z = 0.8
            label.scale.z = 0.4
            label.color.r, label.color.g, label.color.b, label.color.a = 0.0, 0.4, 0.0, 1.0
            label.text = f"FOUND ID{mid}"
            arr.markers.append(label)

        state_label = self._header(Marker())
        state_label.ns, state_label.id = "search_state", 0
        state_label.type, state_label.action = Marker.TEXT_VIEW_FACING, Marker.ADD
        state_label.pose.position.x, state_label.pose.position.y = cx, cy
        state_label.pose.position.z = 1.4
        state_label.scale.z = 0.5
        state_label.color.r, state_label.color.g, state_label.color.b, state_label.color.a = (
            0.8, 0.0, 0.0, 1.0)
        if self.state == STATE_LANDING:
            state_label.text = (
                f"{self.state}/{self.landing_phase} -> ID{self.landing_target_id}")
        else:
            state_label.text = f"{self.state} ({len(self.found_markers)}/{self.n_markers_expected})"
        arr.markers.append(state_label)

        self.viz_pub.publish(arr)

    def _header(self, marker):
        marker.header.frame_id = "world"
        marker.header.stamp = self.get_clock().now().to_msg()
        return marker


def main(args=None):
    rclpy.init(args=args)
    node = SearchPhaseNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

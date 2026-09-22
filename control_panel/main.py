#!/usr/bin/env python3
"""
KASA 정밀착륙 프로젝트 - 통합 제어판

서브시스템(MicroXRCEAgent, usb_cam, aruco_tracker, landing, QGC)을 이 창에서
켜고 끄고, 카메라 라이브 화면·PX4 텔레메트리·오류/경고를 한눈에 봅니다.

실행: run_main.sh 로 실행하세요 (ROS2 환경을 먼저 소싱해줍니다).
"""
import fcntl
import os
import re
import sys
import math
import queue
import signal
import subprocess
import threading
import time
import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox

import numpy as np
import cv2
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from mpl_toolkits.mplot3d.art3d import Line3DCollection

# 기존 검증된 라인트레이서 로직(my_first_pkg)을 그대로 재사용 - 복제 안 함
sys.path.insert(0, "/home/sasllab/ros2_ws/src/my_first_pkg")
from line_tracker import LineTracker
from PIL import Image as PILImage, ImageTk, ImageDraw, ImageFont

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image as RosImage, CameraInfo
from px4_msgs.msg import (
    VehicleAttitude, VehicleLocalPosition, VehicleStatus, BatteryStatus, FailsafeFlags,
    VehicleCommand,
)
from aruco_opencv_msgs.msg import ArucoDetection
from visualization_msgs.msg import MarkerArray

PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

# ---- 자세각 3D 와이어프레임 (attitude_3d_viewer.py 로직 그대로 재사용) ----
# body FRD(X=앞, Y=오른쪽, Z=아래) 기준 십자 프레임 + 노즈(기수) 삼각형.
_ATT_ARM = 1.0
_ATT_BODY_LINES = [
    ([_ATT_ARM, 0, 0], [-_ATT_ARM, 0, 0]),
    ([0, _ATT_ARM, 0], [0, -_ATT_ARM, 0]),
    ([_ATT_ARM, 0, 0], [_ATT_ARM * 0.6, _ATT_ARM * 0.25, 0]),
    ([_ATT_ARM, 0, 0], [_ATT_ARM * 0.6, -_ATT_ARM * 0.25, 0]),
]
_ATT_MOTOR_POINTS_BODY = np.array([
    [_ATT_ARM, 0, 0], [-_ATT_ARM, 0, 0], [0, _ATT_ARM, 0], [0, -_ATT_ARM, 0],
])


def _quat_to_rotmat(w, x, y, z):
    """PX4 q=[w,x,y,z], body(FRD) -> NED 회전행렬."""
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def _ned_to_enu_display(p_ned):
    """NED(N,E,D) -> 화면표시용 (E,N,U): X=동,Y=북,Z=위."""
    n, e, d = p_ned
    return np.array([e, n, -d])

BLUE = "#1c5d7a"        # 상단 바 배경 (기존 유지)
ON_GREEN = "#2e7d32"
OFF_GRAY = "#37474f"

PANEL_BG = "#ffffff"     # cam/log/error 박스 배경 - 화이트 톤
PANEL_BORDER = "#dcdcdc"
ROW_ON_BG = "#e6f4ea"
ROW_ON_BORDER = "#34a853"
ROW_OFF_BG = "#ffffff"
ROW_OFF_BORDER = "#d0d0d0"
ERR_RED = "#d93025"
WARN_AMBER = "#b8860b"

# ---- 켜야 할 것들 ----
SUBSYSTEMS = {
    "xrce": {
        "label": "xrce",
        "cmd": ["MicroXRCEAgent", "serial", "--dev", "/dev/ttyTHS0", "-b", "921600"],
        "danger": False,
    },
    "camera": {
        "label": "camera",
        "cmd": [
            "ros2", "run", "usb_cam", "usb_cam_node_exe", "--ros-args",
            "-p", "video_device:=/dev/video0",
            "-p", "image_width:=1280", "-p", "image_height:=720",
            "-p", "pixel_format:=mjpeg2rgb",
            "-p", "framerate:=30.0",  # C270는 1280x720에서 30Hz가 하드웨어 상한
            "-p", "camera_info_url:=file:///home/sasllab/Desktop/kasa_project/camera_calibration.yaml",
            "-r", "image_raw:=/camera/image_raw",
            "-r", "camera_info:=/camera/camera_info",
        ],
        "danger": False,
    },
    "aruco": {
        "label": "aruco",
        "cmd": [
            "ros2", "run", "aruco_opencv", "aruco_tracker_autostart", "--ros-args",
            "--params-file", "/home/sasllab/PX4-Autopilot/aruco_tracker.yaml",
            "-p", "use_sim_time:=false",
        ],
        "danger": False,
    },
    "qgc": {
        "label": "qgc (Docker)",
        # 호스트 glibc가 낮아 네이티브 AppImage는 GLIBC_2.32+ 없음으로 실행
        # 불가 + 실행 시도 자체가 X11을 죽이는 게 실측됨. 대신 glibc 2.39인
        # ubuntu:24.04 기반 qgc_container(이미 AppImage/X11 소켓 마운트되어
        # 있음)에서 압축해제된 AppRun을 root 아닌 ubuntu 계정으로 실행하면
        # 정상 작동함(root로 실행하면 QGC가 자체적으로 거부하고 종료함).
        "cmd": [
            "docker", "exec", "-u", "ubuntu", "-e", "HOME=/home/ubuntu",
            "qgc_container",
            "/tmp/appimage_extracted_7783564aefad9bb2447305a25068abf6/AppRun",
        ],
        "stop_cmd": ["docker", "exec", "qgc_container", "pkill", "-9", "-f", "QGroundControl"],
        "danger": False,
    },
    "landing": {
        "label": "landing [주의]",
        "cmd": ["ros2", "run", "precision_landing", "landing"],
        "danger": True,
    },
    "mission": {
        "label": "미션(탐색+착륙) [주의]",
        # -p n_markers_expected는 _toggle()에서 GUI 입력값을 보고 실행 직전에
        # 붙임(대회 당일 통보받는 값이라 고정 못 함) - 여기 기본 cmd는 뼈대만.
        "cmd": ["python3", "/home/sasllab/ros2_ws/src/my_first_pkg/search_phase_node.py",
                "--ros-args"],
        "danger": True,
    },
}

# ---- 무해한 것으로 이미 확인된 패턴 -> 오류/경고 패널에 아예 안 띄움 ----
# (2026-09-07 실측: QGC의 정상적인 시작 로그 - TTS 플러그인 없음/오디오
# 없음/지도 타일 확대 경고/영상 위젯 초기화 실패(Docker에 GPU 미할당이라
# 발생하는 정상적인 제약) - 가 전부 "알려지지 않은 오류"로 잘못 분류돼서
# 패널이 도배되는 문제가 있었음. 핵심 기능(비행 제어)엔 영향 없는 것들.)
BENIGN_PATTERNS = [
    re.compile(r"text-to-speech|speechd|QTextToSpeech"),
    re.compile(r"No usable.*AudioOutput"),
    re.compile(r"Bing Tile Above Zoom Level"),
    re.compile(r"video initialization failed|failed to create drawable|VideoManager"),
    re.compile(r"GStreamer"),  # 비디오 스트리밍 플러그인 - MAVLink 핵심 기능과 무관
    re.compile(r"Error loading (source|json) localization"),  # "C" 로케일 관련, 무해
    re.compile(r"QQuickPinchArea"),  # QML 프로퍼티 오버라이드 경고, 동작에 영향 없음
]

# ---- 알려진 오류 패턴 -> (심각도, 설명/해결법) ----
KNOWN_ISSUES = [
    (re.compile(r"GLIBC_"), "error", "QGC 실행 실패: 이 Jetson glibc 버전이 낮아 최신 QGC AppImage 미지원. 파라미터는 pymavlink로 관리."),
    (re.compile(r"convertToGrey|cv::Exception"), "error", "카메라 인코딩이 OpenCV와 안 맞음. usb_cam pixel_format=mjpeg2rgb 확인."),
    (re.compile(r"command \d+ unsupported"), "error", "PX4 commander가 해당 MAVLink 명령 미지원. actuator_test 콘솔 명령 사용 권장."),
    (re.compile(r"No such file or directory.*ttyACM|could not open port"), "error", "USB 연결 끊김. 케이블/포트 확인 필요."),
    (re.compile(r"disconnected"), "warn", "PX4 DDS 링크 끊김 가능성. Agent/배선 상태 확인."),
    (re.compile(r"Preflight Fail"), "warn", "PX4 프리플라이트 체크 경고 (배터리/자세/자기간섭 등). 비행 전 원인 해소 필요."),
]


def classify_line(line):
    for pattern in BENIGN_PATTERNS:
        if pattern.search(line):
            return None
    for pattern, sev, hint in KNOWN_ISSUES:
        if pattern.search(line):
            return sev, hint
    low = line.lower()
    # QGC는 cosmetic한 문제도 "Critical:"로 로깅해서 critical 단어만으로
    # 걸면 노이즈가 너무 많음(2026-09-07 실측) -> traceback/error만 트리거.
    if "traceback" in low or low.startswith("error") or " error" in low:
        return "error", "확인 필요 (알려지지 않은 오류) - 로그 참고"
    if "warn" in low:
        return "warn", "확인 필요 (경고) - 로그 참고"
    return None


class Subsystem:
    def __init__(self, key, cfg, log_q):
        self.key = key
        self.cfg = cfg
        self.log_q = log_q
        self.proc = None
        self.thread = None

    def is_running(self):
        return self.proc is not None and self.proc.poll() is None

    def start(self):
        if self.is_running():
            return
        env = os.environ.copy()
        env.setdefault("ROS_DOMAIN_ID", "0")
        env.setdefault("DISPLAY", ":1")
        try:
            self.proc = subprocess.Popen(
                self.cfg["cmd"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1, env=env, start_new_session=True,
            )
        except FileNotFoundError as e:
            self.log_q.put((self.key, f"[실행 실패] {e}"))
            return
        self.thread = threading.Thread(target=self._reader, daemon=True)
        self.thread.start()

    def stop(self):
        if self.proc is None:
            return
        # cmd가 "ros2 run ..."인 경우, 그 프로세스는 실제 노드 실행파일을
        # 자식으로 fork/exec하고 자신은 안 죽는 경우가 있어(ros2 CLI 래퍼가
        # 시그널을 자식으로 전달 안 함 - usb_cam/aruco_tracker에서 실측됨:
        # proc.terminate()로 래퍼만 죽고 실제 노드는 살아남아 /dev/video0를
        # 계속 붙잡고 있어서 다음 재시작이 crash함) - start_new_session=True로
        # 만든 프로세스 그룹 전체에 시그널을 보내야 진짜 자식까지 다 죽는다.
        try:
            pgid = os.getpgid(self.proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            self.proc.wait(timeout=3)
        except Exception:
            try:
                os.killpg(pgid, signal.SIGKILL)
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
        self.proc = None

        # docker exec처럼, 로컬 클라이언트 프로세스를 죽여도 컨테이너 안
        # 프로세스는 안 죽는 경우가 있어 별도 정리 명령이 필요할 수 있다.
        stop_cmd = self.cfg.get("stop_cmd")
        if stop_cmd:
            try:
                subprocess.run(stop_cmd, timeout=5, capture_output=True)
            except Exception:
                pass

    def _reader(self):
        # 예전엔 여기서 N줄 넘으면 읽기를 break로 중단시켰는데, 그러면
        # OS 파이프 버퍼가 다 차서 자식 프로세스가 자기 stdout write()에서
        # 블록되어 그대로 멈춰버리는 문제가 있었다(실측). 그래서 프로세스가
        # 살아있는 한 끝까지 계속 읽어서 파이프를 비워준다 - 로그를
        # 생략하지 않고, 프로세스도 안 멈추게. (X RENDER 크래시의 진짜
        # 원인은 컬러 이모지 글리프였고 이미 제거됨 - 로그 줄 수 자체는
        # 문제가 아니었다.)
        proc = self.proc
        try:
            for line in proc.stdout:
                self.log_q.put((self.key, line.rstrip("\n")))
        except Exception:
            pass
        self.log_q.put((self.key, "[프로세스 종료됨]"))


class DirectCamCapture:
    """ROS2/usb_cam 없이 cv2.VideoCapture로 바로 웹캠을 열어 tele_state에
    프레임을 채워 넣는 테스트용 캡처 스레드. 기존 카메라 표시/오버레이
    코드(_refresh_camera)는 frame 출처를 신경 안 쓰므로 그대로 재사용된다."""

    def __init__(self, dev, state, lock):
        self.dev = dev
        self.state = state
        self.lock = lock
        self.cap = None
        self.thread = None
        self.running = False

    def start(self):
        if self.running:
            return
        cap = cv2.VideoCapture(self.dev)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        if not cap.isOpened():
            raise RuntimeError(f"{self.dev} 열기 실패 (다른 프로세스가 점유 중일 수 있음)")
        self.cap = cap
        self.running = True
        self.thread = threading.Thread(target=self._loop, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False
        if self.thread:
            self.thread.join(timeout=2)
            self.thread = None
        if self.cap:
            self.cap.release()
            self.cap = None
        with self.lock:
            self.state.pop("frame", None)
            self.state.pop("frame_t", None)

    def _loop(self):
        while self.running and self.cap is not None:
            ok, frame_bgr = self.cap.read()
            if not ok:
                time.sleep(0.05)
                continue
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            with self.lock:
                self.state["frame"] = frame_rgb
                self.state["frame_t"] = time.time()
            time.sleep(0.03)


class TelemetryNode(Node):
    def __init__(self, state, lock):
        super().__init__("kasa_control_panel")
        self.state = state
        self.lock = lock
        self.create_subscription(VehicleAttitude, "/fmu/out/vehicle_attitude", self._att, PX4_QOS)
        # 2026-09-21 실측: 이 PX4 빌드는 vehicle_local_position/battery_status를
        # 버전 없는 이름으로는 아예 안 내보내고 "_v1" 이름으로만 퍼블리시함
        # (ros2 topic info로 확인: 버전 없는 쪽 Publisher count 0, _v1 쪽 1).
        self.create_subscription(VehicleLocalPosition, "/fmu/out/vehicle_local_position_v1", self._pos, PX4_QOS)
        self.create_subscription(VehicleStatus, "/fmu/out/vehicle_status_v1", self._status, PX4_QOS)
        self.create_subscription(BatteryStatus, "/fmu/out/battery_status_v1", self._batt, PX4_QOS)
        self.create_subscription(ArucoDetection, "/aruco_detections", self._aruco, 10)
        self.create_subscription(RosImage, "/camera/image_raw", self._cam, 5)
        self.create_subscription(CameraInfo, "/camera/camera_info", self._cam_info, 5)
        self.create_subscription(MarkerArray, "/kasa_search_markers", self._mission, 10)
        self.create_subscription(FailsafeFlags, "/fmu/out/failsafe_flags", self._failsafe, PX4_QOS)
        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", PX4_QOS)

    def _set(self, key, value):
        with self.lock:
            self.state[key] = value
            self.state[key + "_t"] = time.time()

    def _att(self, msg):
        w, x, y, z = msg.q
        n = (w * w + x * x + y * y + z * z) ** 0.5
        if n < 1e-9:
            return
        w, x, y, z = w / n, x / n, y / n, z / n
        roll = np.degrees(np.arctan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)))
        sinp = max(-1.0, min(1.0, 2 * (w * y - z * x)))
        pitch = np.degrees(np.arcsin(sinp))
        yaw = np.degrees(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
        self._set("rpy", (roll, pitch, yaw))
        self._set("quat", (w, x, y, z))

    def _pos(self, msg):
        self._set("pos", (msg.x, msg.y, msg.z))
        # 상대고도: 거리센서(dist_bottom)가 유효하면 그게 더 정확(지형 추종),
        # 아니면 로컬 원점 기준 z(NED, 아래가 +)를 뒤집어서 대체.
        rel_alt = msg.dist_bottom if msg.dist_bottom_valid else (-msg.z if msg.z_valid else None)
        self._set("flight", (msg.vx, msg.vy, msg.vz, rel_alt))

    def _status(self, msg):
        self._set("armed", msg.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        self._set("nav_state", msg.nav_state)

    def _failsafe(self, msg):
        self._set("safety", (
            msg.local_position_invalid, msg.global_position_invalid, msg.home_position_invalid,
        ))

    def _batt(self, msg):
        self._set("battery", (msg.voltage_v, msg.remaining, msg.warning))

    def _aruco(self, msg):
        ids = [m.marker_id for m in msg.markers]
        self._set("markers", ids)
        infos = []
        poses = []
        for m in msg.markers:
            p = m.pose.position
            q = m.pose.orientation
            dist = (p.x ** 2 + p.y ** 2 + p.z ** 2) ** 0.5
            infos.append((m.marker_id, dist))
            poses.append((m.marker_id, p.x, p.y, p.z, q.x, q.y, q.z, q.w))
        self._set("marker_info", infos)
        self._set("marker_poses", poses)

    def _cam_info(self, msg):
        # XYZ 축 오버레이(cv2.drawFrameAxes)용 카메라 내부파라미터.
        K = np.array(msg.k, dtype=np.float64).reshape(3, 3)
        D = np.array(msg.d, dtype=np.float64)
        if K[0, 0] == 0:
            return  # usb_cam이 아직 캘리브레이션 안 된 기본값(전부 0)을 보낸 경우 무시
        self._set("cam_K", K)
        self._set("cam_D", D)

    def _cam(self, msg):
        if msg.encoding != "rgb8":
            return
        arr = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, 3)
        self._set("frame", arr.copy())

    def _mission(self, msg):
        for marker in msg.markers:
            if marker.ns == "search_state":
                self._set("mission_state", marker.text)
                return

    def reboot_pixhawk(self):
        # UXRCE-DDS 세션이 죽은 뒤 젯슨 쪽만 재시작하면 PX4가 알아서 재연결
        # 안 하는 경우가 잦아서(2026-09-08 실측, 매번 전원 재인가로만 풀림)
        # 물리적으로 전원을 뽑았다 꽂는 대신 이 명령으로 대신한다. PX4는
        # armed 상태에선 이 명령을 자체적으로 거부함(안전장치 내장).
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = VehicleCommand.VEHICLE_CMD_PREFLIGHT_REBOOT_SHUTDOWN
        msg.param1 = 1.0  # 1: 오토파일럿만 재부팅
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)


class App:
    def __init__(self, root):
        self.root = root
        root.title("KASA 정밀착륙 제어판")
        root.geometry("1150x850")
        root.configure(bg="white")

        self.CAM_DISPLAY_SIZE = (640, 360)
        # 서브시스템/오류 로그가 지금까지 Text 위젯(인메모리)에만 쌓여서 창을
        # 닫으면 다 사라졌음(2026-09-08 실측 지적) - 세션마다 새 파일에
        # append해서 나중에 "그때 왜 이상했지" 복기가 가능하게 함.
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"kasa_{time.strftime('%Y%m%d_%H%M%S')}.log")
        self.log_file = open(log_path, "a", encoding="utf-8")
        self.log_file.write(f"=== KASA 정밀착륙 제어판 로그 시작 {time.strftime('%Y-%m-%d %H:%M:%S')} ===\n")
        self.log_file.flush()
        self.log_q = queue.Queue()
        self.subsystems = {k: Subsystem(k, cfg, self.log_q) for k, cfg in SUBSYSTEMS.items()}
        self.tele_state = {}
        self.tele_lock = threading.Lock()
        self.issues = []  # (severity, text)
        self.direct_cam = DirectCamCapture("/dev/video0", self.tele_state, self.tele_lock)
        self._prev_running = {}   # 토글 버튼 불필요한 재렌더링 방지용
        self._prev_pill = {}      # 상단 pill 불필요한 재렌더링 방지용

        # 체스보드 캘리브레이션 상태
        self.calib_objpoints = []
        self.calib_imgpoints = []
        self.calib_last_corners = None
        self.calib_last_pattern = None
        self.calib_last_image_size = None
        self.CALIB_SAVE_PATH = "/home/sasllab/Desktop/kasa_project/camera_calibration.yaml"

        self.line_tracker = LineTracker()

        self._build_ui()

        # xrce는 켤 때마다 수동으로 눌러줘야 하는 게 번거롭다는 실사용 피드백
        # (2026-09-22) -> GUI 뜰 때 기본 포트/baud로 자동 시작. PX4쪽
        # uxrce_dds_client 자동시작 여부와는 무관한 별개 계층(Jetson agent)이라
        # 이거 하나 켠다고 PX4쪽까지 다 해결되진 않음 - 안 붙으면 여전히 PX4
        # 셸에서 uxrce_dds_client 재시작 필요.
        self._toggle("xrce")

        rclpy.init()
        self.node = TelemetryNode(self.tele_state, self.tele_lock)
        self.spin_thread = threading.Thread(target=rclpy.spin, args=(self.node,), daemon=True)
        self.spin_thread.start()

        self.root.after(150, self._poll_logs)
        self.root.after(100, self._refresh_telemetry)
        self.root.after(50, self._refresh_camera)
        root.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------------- UI ----------------
    def _build_ui(self):
        # ---- 상단 상태 바 ----
        top = tk.Frame(self.root, bg=BLUE)
        top.pack(fill=tk.X)
        self.top_pills = {}
        pill_keys = list(SUBSYSTEMS.keys()) + ["px4"]
        for key in pill_keys:
            label = "PX4" if key == "px4" else key
            f = tk.Frame(top, bg=OFF_GRAY, padx=8, pady=6)
            f.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=1, pady=1)
            lbl = tk.Label(f, text=label, bg=OFF_GRAY, fg="white", font=("", 10, "bold"))
            lbl.pack()
            f.pill_label = lbl
            self.top_pills[key] = f

        # ---- RXTX UART 통신 점검용 실시간 자세각 ----
        # PX4가 자세를 계속 발행해야 갱신되므로, 기체를 손으로 기울였을 때 이
        # 숫자가 실시간으로 따라 바뀌는지 보면 배선/UART/uXRCE-DDS 링크가
        # 실제로 살아있는지 눈으로 바로 확인 가능 (px4 핀은 "최근 2초 내
        # nav_state 수신"만 보므로, 자세 갱신이 끊겨도 한동안 초록으로 남아
        # 있을 수 있어 이걸로 보완).
        att_bar = tk.Frame(self.root, bg=PANEL_BG, highlightbackground=PANEL_BORDER, highlightthickness=1)
        att_bar.pack(fill=tk.X)
        tk.Label(att_bar, text="자세각(UART 점검):", bg=PANEL_BG, fg="#333333",
                 font=("", 9, "bold")).pack(side=tk.LEFT, padx=(10, 6), pady=4)
        # width=고정 + monospace라, 값이 바뀌어도(자릿수/부호 달라져도) 옆 위젯이
        # 좌우로 안 밀림 - 전에는 텍스트 길이대로 라벨이 늘었다 줄었다 해서
        # 그 오른쪽 위젯들이 계속 옆으로 움직였음(실측 지적됨).
        self.att_label = tk.Label(att_bar, text="Roll ─   Pitch ─   Yaw ─", bg=PANEL_BG,
                                   fg="#999999", font=("monospace", 10), width=34, anchor="w")
        self.att_label.pack(side=tk.LEFT, pady=4)
        self.att_age_label = tk.Label(att_bar, text="", bg=PANEL_BG, fg="#999999",
                                       font=("monospace", 9), width=14, anchor="w")
        self.att_age_label.pack(side=tk.LEFT, padx=10, pady=4)
        self.batt_label = tk.Label(att_bar, text="배터리 ─", bg=PANEL_BG, fg="#999999",
                                    font=("monospace", 10, "bold"), width=16, anchor="w")
        self.batt_label.pack(side=tk.LEFT, padx=(20, 10), pady=4)
        # xrce 에이전트만 재시작해서는 PX4가 재연결 안 하는 경우가 잦아서
        # (2026-09-08 실측, 매번 전원 재인가로만 풀렸음) 물리적 리부팅 대신
        # VEHICLE_CMD_PREFLIGHT_REBOOT_SHUTDOWN을 보내는 버튼 - armed 중엔
        # PX4가 이 명령 자체를 거부하니 위험하지 않음.
        tk.Button(att_bar, text="Pixhawk 재시작", command=self._reboot_pixhawk,
                  bg=ROW_OFF_BG, relief="solid", bd=1, font=("", 9), padx=8,
                  cursor="hand2").pack(side=tk.RIGHT, padx=(0, 10), pady=4)

        # attitude_3d_viewer.py와 동일한 와이어프레임 - 손으로 기체를 기울였을
        # 때 3D 모형이 실시간으로 따라 도는지 눈으로 바로 확인 가능.
        att_fig = Figure(figsize=(1.0, 1.0), dpi=72)
        self.att_ax = att_fig.add_subplot(111, projection="3d")
        att_fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        for axis_label, lim_setter in (
                ("East", self.att_ax.set_xlim), ("North", self.att_ax.set_ylim), ("Up", self.att_ax.set_zlim)):
            lim_setter(-1.5, 1.5)
        self.att_ax.set_xticks([]); self.att_ax.set_yticks([]); self.att_ax.set_zticks([])
        self.att_lc = Line3DCollection([], colors="k", linewidths=2)
        self.att_ax.add_collection3d(self.att_lc)
        self.att_nose_lc = Line3DCollection([], colors="r", linewidths=3)
        self.att_ax.add_collection3d(self.att_nose_lc)
        self.att_motor_scatter = self.att_ax.scatter([], [], [], color="b", s=30)
        self.att_canvas = FigureCanvasTkAgg(att_fig, master=att_bar)
        self.att_canvas.get_tk_widget().pack(side=tk.RIGHT, padx=(0, 8), pady=2)
        self._draw_attitude_3d(1.0, 0.0, 0.0, 0.0)  # 초기 자세(수평)로 한 번 그려둠

        body = tk.Frame(self.root, bg="white")
        body.pack(fill=tk.BOTH, expand=True)

        left = tk.Frame(body, bg="white", width=230)
        left.pack(side=tk.LEFT, fill=tk.Y)
        left.pack_propagate(False)

        sep = tk.Frame(body, bg=BLUE, width=2)
        sep.pack(side=tk.LEFT, fill=tk.Y)

        right = tk.Frame(body, bg="white")
        right.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        # ---- 왼쪽 위: 켜야할 것들 ----
        tk.Label(left, text="켜야할 것들", bg=BLUE, fg="white", font=("", 11, "bold"), anchor="w").pack(fill=tk.X, ipady=4)
        self.toggle_rows = {}
        for key, cfg in SUBSYSTEMS.items():
            disabled = bool(cfg.get("disabled_reason"))
            dot = "X" if disabled else "○"
            btn = tk.Button(
                left, text=f"{dot}  {cfg['label']}", anchor="w", justify="left",
                bg=ROW_OFF_BG, fg="#999999" if disabled else "#333333", activebackground=ROW_OFF_BG,
                relief="solid", bd=1, highlightbackground=ROW_OFF_BORDER,
                font=("", 10), padx=10, pady=8, cursor="hand2",
                command=lambda k=key: self._toggle(k),
            )
            btn.pack(fill=tk.X, padx=6, pady=3)
            self.toggle_rows[key] = btn
            if key == "mission":
                # 대회 규정상 목표 마커 개수는 당일 통보값이라 하드코딩 불가
                # (search_phase_node.py의 n_markers_expected 파라미터로 감).
                mrow = tk.Frame(left, bg="white")
                mrow.pack(fill=tk.X, padx=6, pady=(0, 3))
                tk.Label(mrow, text="목표 마커 개수:", bg="white", font=("", 9)).pack(side=tk.LEFT)
                self.mission_markers_var = tk.StringVar(value="4")
                tk.Entry(mrow, textvariable=self.mission_markers_var, width=4).pack(side=tk.LEFT, padx=(4, 0))
            if key == "xrce":
                # Pixhawk와 물린 포트가 배선/USB 재연결에 따라 바뀔 수 있어
                # (/dev/ttyTHS0=Jetson 40핀 헤더 UART, /dev/pixhawk=USB 심볼릭
                # 링크) 실행 직전에 GUI에서 고르게 함 - xrce 하나만 하드코딩된
                # /dev/ttyTHS0라 매번 코드 고쳐야 했던 문제(실측) 해결.
                xrow = tk.Frame(left, bg="white")
                xrow.pack(fill=tk.X, padx=6, pady=(0, 3))
                tk.Label(xrow, text="포트:", bg="white", font=("", 9)).pack(side=tk.LEFT)
                self.xrce_port_var = tk.StringVar(value="/dev/ttyTHS0")
                ttk.Combobox(
                    xrow, textvariable=self.xrce_port_var, width=14,
                    values=["/dev/ttyTHS0", "/dev/pixhawk", "/dev/ttyACM0"],
                ).pack(side=tk.LEFT, padx=(4, 0))
                xbrow = tk.Frame(left, bg="white")
                xbrow.pack(fill=tk.X, padx=6, pady=(0, 3))
                tk.Label(xbrow, text="baud:", bg="white", font=("", 9)).pack(side=tk.LEFT)
                self.xrce_baud_var = tk.StringVar(value="921600")
                ttk.Combobox(
                    xbrow, textvariable=self.xrce_baud_var, width=8,
                    values=["921600", "460800", "230400", "115200", "57600"],
                ).pack(side=tk.LEFT, padx=(4, 0))

        # ---- 왼쪽 맨 아래: 비행 정보(속도/상대고도) ----
        # side=BOTTOM으로 먼저 pack해서 아래쪽에 고정 - 그래야 나중에 pack하는
        # 오류 목록 박스(expand=True)가 이 위 남은 공간만 채우고, 이 박스를
        # 밀어내지 않는다.
        flight_box = tk.Frame(left, bg=PANEL_BG, highlightbackground=PANEL_BORDER, highlightthickness=1)
        flight_box.pack(side=tk.BOTTOM, fill=tk.X, padx=6, pady=6)
        tk.Label(flight_box, text="비행 정보", bg=BLUE, fg="white", font=("", 10, "bold"), anchor="w").pack(fill=tk.X, ipady=3)
        flight_rows = tk.Frame(flight_box, bg=PANEL_BG)
        flight_rows.pack(fill=tk.X, padx=8, pady=6)
        self.flight_relalt_label = tk.Label(flight_rows, text="상대고도: ─", bg=PANEL_BG, fg="#999999",
                                             font=("monospace", 9), anchor="w")
        self.flight_relalt_label.pack(fill=tk.X)
        self.flight_hspeed_label = tk.Label(flight_rows, text="수평속도: ─", bg=PANEL_BG, fg="#999999",
                                             font=("monospace", 9), anchor="w")
        self.flight_hspeed_label.pack(fill=tk.X)
        self.flight_vspeed_label = tk.Label(flight_rows, text="수직속도: ─", bg=PANEL_BG, fg="#999999",
                                             font=("monospace", 9), anchor="w")
        self.flight_vspeed_label.pack(fill=tk.X)
        # search_phase_node.py가 이미 쏘고 있는 /kasa_search_markers(RViz용)를
        # 그대로 재사용 - 새 통신 경로 안 만들고 "search_state" ns 마커의
        # text 필드만 읽어옴(state_label 텍스트, 08 탐색 페이즈 노드 참고).
        self.mission_state_label = tk.Label(flight_rows, text="미션: ─", bg=PANEL_BG, fg="#999999",
                                             font=("monospace", 9), anchor="w")
        self.mission_state_label.pack(fill=tk.X)
        # ---- pre-arm/안전 상태 (arming + 위치추정 유효성) ----
        self.arm_state_label = tk.Label(flight_rows, text="ARM: ─", bg=PANEL_BG, fg="#999999",
                                         font=("monospace", 9, "bold"), anchor="w")
        self.arm_state_label.pack(fill=tk.X)
        self.safety_label = tk.Label(flight_rows, text="위치추정: ─", bg=PANEL_BG, fg="#999999",
                                      font=("monospace", 9), anchor="w")
        self.safety_label.pack(fill=tk.X)

        # ---- 왼쪽 아래: 오류/경고 ----
        err_box = tk.Frame(left, bg=PANEL_BG, highlightbackground=PANEL_BORDER, highlightthickness=1)
        err_box.pack(fill=tk.BOTH, expand=True, padx=6, pady=(20, 6))
        tk.Label(err_box, text="오류 목록 / 해결 / 경고", bg=BLUE, fg="white", font=("", 10, "bold"), anchor="w").pack(fill=tk.X, ipady=3)
        self.err_text = tk.Text(err_box, bg=PANEL_BG, fg="#333333", font=("", 9), wrap=tk.WORD,
                                 borderwidth=0, highlightthickness=0)
        self.err_text.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        self.err_text.tag_configure("error", foreground=ERR_RED)
        self.err_text.tag_configure("warn", foreground=WARN_AMBER)

        # ---- 오른쪽 위: 카메라 ----
        cam_mode_row = tk.Frame(right, bg="white")
        cam_mode_row.pack(fill=tk.X, padx=10, pady=(10, 0))
        self.cam_mode = tk.StringVar(value="normal")
        tk.Radiobutton(cam_mode_row, text="일반", variable=self.cam_mode, value="normal",
                       bg="white", font=("", 9)).pack(side=tk.LEFT)
        tk.Radiobutton(cam_mode_row, text="패드 경계 감지", variable=self.cam_mode, value="boundary",
                       bg="white", font=("", 9)).pack(side=tk.LEFT, padx=(10, 0))
        tk.Radiobutton(cam_mode_row, text="라인트레이서(그리드)", variable=self.cam_mode, value="linetrack",
                       bg="white", font=("", 9)).pack(side=tk.LEFT, padx=(10, 0))
        tk.Radiobutton(cam_mode_row, text="캘리브레이션", variable=self.cam_mode, value="calib",
                       bg="white", font=("", 9)).pack(side=tk.LEFT, padx=(10, 0))
        self.direct_cam_btn = tk.Button(
            cam_mode_row, text="웹캠 직접 테스트 시작", command=self._toggle_direct_cam,
            bg=ROW_OFF_BG, relief="solid", bd=1, highlightbackground=ROW_OFF_BORDER,
            font=("", 9), padx=8, cursor="hand2",
        )
        self.direct_cam_btn.pack(side=tk.RIGHT)

        # ---- 패드 경계 감지 튜닝 ----
        boundary_row = tk.Frame(right, bg="white")
        boundary_row.pack(fill=tk.X, padx=10, pady=(4, 0))
        tk.Label(boundary_row, text="패드 경계 최소 크기(전체화면 대비 %)", bg="white", font=("", 8)).pack(side=tk.LEFT)
        self.boundary_min_area_var = tk.StringVar(value="1.5")
        tk.Entry(boundary_row, textvariable=self.boundary_min_area_var, width=5).pack(side=tk.LEFT, padx=(4, 0))
        tk.Label(boundary_row, text="(패드가 화면에서 작게 보이면 값을 줄이세요)",
                 bg="white", fg="#888888", font=("", 8)).pack(side=tk.LEFT, padx=(8, 0))

        # ---- 캘리브레이션 컨트롤 ----
        calib_row = tk.Frame(right, bg="white")
        calib_row.pack(fill=tk.X, padx=10, pady=(4, 0))
        tk.Label(calib_row, text="체스보드 내부코너(가로x세로)", bg="white", font=("", 8)).pack(side=tk.LEFT)
        self.calib_cols_var = tk.StringVar(value="9")
        tk.Entry(calib_row, textvariable=self.calib_cols_var, width=3).pack(side=tk.LEFT, padx=(4, 0))
        tk.Label(calib_row, text="x", bg="white", font=("", 8)).pack(side=tk.LEFT)
        self.calib_rows_var = tk.StringVar(value="6")
        tk.Entry(calib_row, textvariable=self.calib_rows_var, width=3).pack(side=tk.LEFT, padx=(0, 8))
        tk.Label(calib_row, text="칸 크기(mm)", bg="white", font=("", 8)).pack(side=tk.LEFT)
        self.calib_square_mm_var = tk.StringVar(value="25")
        tk.Entry(calib_row, textvariable=self.calib_square_mm_var, width=5).pack(side=tk.LEFT, padx=(4, 12))
        self.calib_sample_label = tk.Label(calib_row, text="샘플: 0", bg="white", font=("", 9, "bold"))
        self.calib_sample_label.pack(side=tk.LEFT, padx=(0, 12))
        tk.Button(calib_row, text="캡처", command=self._calib_capture,
                  bg=ROW_OFF_BG, relief="solid", bd=1, font=("", 9), padx=6, cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(calib_row, text="계산+저장", command=self._calib_compute,
                  bg=ROW_OFF_BG, relief="solid", bd=1, font=("", 9), padx=6, cursor="hand2").pack(side=tk.LEFT, padx=2)
        tk.Button(calib_row, text="초기화", command=self._calib_reset,
                  bg=ROW_OFF_BG, relief="solid", bd=1, font=("", 9), padx=6, cursor="hand2").pack(side=tk.LEFT, padx=2)
        self.flip_var = tk.BooleanVar(value=False)
        tk.Checkbutton(calib_row, text="좌우반전", variable=self.flip_var,
                       bg="white", font=("", 9)).pack(side=tk.LEFT, padx=(12, 2))

        cam_box = tk.Frame(right, bg=PANEL_BG, highlightbackground=PANEL_BORDER, highlightthickness=1,
                            width=760, height=440)
        cam_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(4, 5))
        cam_box.pack_propagate(False)  # cam_label(이미지) 크기가 이 박스 크기에 영향 못 주게 고정
        self.cam_box = cam_box
        self.cam_label = tk.Label(cam_box, text="cam\ncamera를 켜세요", bg=PANEL_BG, fg="#888888", font=("", 12))
        self.cam_label.pack(expand=True)

        # ---- 오른쪽 아래: 로그 ----
        log_box = tk.Frame(right, bg=PANEL_BG, highlightbackground=PANEL_BORDER, highlightthickness=1)
        log_box.pack(fill=tk.BOTH, expand=True, padx=10, pady=(5, 10))
        self.log_text = scrolledtext.ScrolledText(log_box, wrap=tk.WORD, font=("monospace", 9),
                                                   bg=PANEL_BG, fg="#333333", insertbackground="black",
                                                   borderwidth=0, highlightthickness=0)
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=2, pady=2)

    # ---------------- 서브시스템 제어 ----------------
    def _toggle(self, key):
        disabled_reason = SUBSYSTEMS[key].get("disabled_reason")
        if disabled_reason:
            messagebox.showwarning(f"{key} 실행 불가", disabled_reason)
            return

        sub = self.subsystems[key]
        if sub.is_running():
            sub.stop()
        else:
            if key == "mission":
                n = self.mission_markers_var.get().strip()
                if not n.isdigit():
                    messagebox.showwarning("미션 시작 불가", "목표 마커 개수를 숫자로 입력하세요.")
                    return
                # 대회 당일 통보받는 값이라 실행 직전 GUI 입력값을 그대로 붙임
                # (SUBSYSTEMS 딕셔너리는 Subsystem이 그대로 참조하고 있어서
                # 여기서 갱신하면 sub.start()가 바로 이 값을 씀).
                base_cmd = ["python3", "/home/sasllab/ros2_ws/src/my_first_pkg/search_phase_node.py",
                            "--ros-args", "-p", f"n_markers_expected:={n}"]
                SUBSYSTEMS["mission"]["cmd"] = base_cmd
            if key == "xrce":
                port = self.xrce_port_var.get().strip()
                baud = self.xrce_baud_var.get().strip()
                if not port:
                    messagebox.showwarning("xrce 시작 불가", "포트를 입력하세요.")
                    return
                if not baud.isdigit():
                    messagebox.showwarning("xrce 시작 불가", "baud를 숫자로 입력하세요.")
                    return
                SUBSYSTEMS["xrce"]["cmd"] = ["MicroXRCEAgent", "serial", "--dev", port, "-b", baud]
            if SUBSYSTEMS[key]["danger"]:
                msg = (
                    "탐색+착륙 미션 노드는 실제로 이륙·자율비행·착륙까지 전부 자동으로 진행합니다.\n"
                    "프로펠러 장착 여부와 주변 안전, 비행 공간 확보를 반드시 확인했나요?"
                    if key == "mission" else
                    "landing 노드는 실제로 Offboard 제어 명령을 PX4로 보냅니다.\n"
                    "프로펠러 장착 여부와 주변 안전을 반드시 확인했나요?"
                )
                ok = messagebox.askyesno("주의", msg)
                if not ok:
                    return
            if key == "camera" and self.direct_cam.running:
                # /dev/video0를 direct_cam이 이미 물고 있으면 usb_cam이 못 여니 먼저 끔
                self._toggle_direct_cam()
            sub.start()

    def _reboot_pixhawk(self):
        ok = messagebox.askyesno(
            "Pixhawk 재시작",
            "Pixhawk에 재부팅 명령을 보냅니다. (armed 상태면 PX4가 자체적으로 거부함)\n계속할까요?",
        )
        if not ok:
            return
        self.node.reboot_pixhawk()
        self._append_log("[reboot] Pixhawk 재부팅 명령 전송됨\n")

    def _write_log_file(self, text):
        try:
            self.log_file.write(text if text.endswith("\n") else text + "\n")
            self.log_file.flush()
        except Exception:
            pass  # 디스크 로그는 부가기능이라, 실패해도 GUI 자체는 계속 동작해야 함

    def _append_log(self, text):
        self.log_text.insert(tk.END, text)
        self.log_text.see(tk.END)
        self._write_log_file(text)

    def _toggle_direct_cam(self):
        if self.direct_cam.running:
            self.direct_cam.stop()
            self.direct_cam_btn.configure(text="웹캠 직접 테스트 시작", bg=ROW_OFF_BG,
                                           highlightbackground=ROW_OFF_BORDER)
            return

        if self.subsystems["camera"].is_running():
            # 서로 /dev/video0를 동시에 못 열므로 usb_cam을 먼저 끔
            self.subsystems["camera"].stop()

        try:
            self.direct_cam.start()
        except Exception as e:
            self._append_log(f"[direct_cam] {e}\n")
            self.err_text.insert(tk.END, f"[ERR] [direct_cam] 웹캠 열기 실패: {e}\n", "error")
            self.err_text.see(tk.END)
            self._write_log_file(f"[ERR] [direct_cam] 웹캠 열기 실패: {e}")
            return

        self.direct_cam_btn.configure(text="웹캠 직접 테스트 중지 (ROS 안 씀)", bg=ROW_ON_BG,
                                       highlightbackground=ROW_ON_BORDER)

    # ---------------- 주기 갱신 ----------------
    def _poll_logs(self):
        # 큐에 한꺼번에 수십 줄이 쌓여있어도(예: qgc GLIBC 에러 덤프) 한 틱에
        # 다 밀어넣으면 X RENDER 확장이 죽는 게 실측됨(BadLength) -> 틱당
        # 처리 개수를 제한해서 여러 틱에 걸쳐 나눠 그리게 한다.
        MAX_LINES_PER_TICK = 5
        processed = 0
        try:
            while processed < MAX_LINES_PER_TICK:
                key, line = self.log_q.get_nowait()
                line = line[:300]  # 비정상적으로 긴 한 줄도 렌더링 부담을 줄이기 위해 자름
                # 파이프는 계속 다 읽어서 프로세스가 안 멈추게 하되(중요),
                # 이미 무해하다고 확인된 반복 스팸(Bing Tile 등)은 로그
                # 화면에도 안 보이게 생략한다 (사용자 요청, 2026-09-07).
                is_noise = any(p.search(line) for p in BENIGN_PATTERNS)
                if not is_noise:
                    self._append_log(f"[{key}] {line}\n")
                result = classify_line(line)
                if result:
                    sev, hint = result
                    tag = "error" if sev == "error" else "warn"
                    prefix = "[ERR]" if sev == "error" else "[WARN]"
                    self.err_text.insert(tk.END, f"{prefix} [{key}] {hint}\n", tag)
                    self.err_text.see(tk.END)
                    self._write_log_file(f"{prefix} [{key}] {hint}")
                processed += 1
        except queue.Empty:
            pass

        for widget in (self.log_text, self.err_text):
            n_lines = int(widget.index("end-1c").split(".")[0])
            if n_lines > 1000:
                widget.delete("1.0", f"{n_lines - 500}.0")

        # X11/Tk에서 위젯을 매 폴링(150ms)마다 재구성하면 폰트 렌더링(RENDER
        # 확장) 요청이 누적되어 오래 켜두면 "BadLength RenderAddGlyphs"로
        # 죽는 게 실측됨 -> 상태가 실제로 바뀔 때만 configure() 호출.
        data_driven = {"camera", "aruco"}
        for key, sub in self.subsystems.items():
            running = sub.is_running()
            if self._prev_running.get(key) == running:
                continue
            self._prev_running[key] = running
            btn = self.toggle_rows[key]
            dot = "●" if running else "○"
            btn.configure(
                text=f"{dot}  {SUBSYSTEMS[key]['label']}",
                bg=ROW_ON_BG if running else ROW_OFF_BG,
                activebackground=ROW_ON_BG if running else ROW_OFF_BG,
                fg="#1e7e34" if running else "#333333",
                highlightbackground=ROW_ON_BORDER if running else ROW_OFF_BORDER,
            )
            if key not in data_driven:
                self._set_pill(key, running)

        self.root.after(150, self._poll_logs)

    def _set_pill(self, key, on):
        if self._prev_pill.get(key) == on:
            return
        self._prev_pill[key] = on
        f = self.top_pills.get(key)
        if not f:
            return
        color = ON_GREEN if on else OFF_GRAY
        f.configure(bg=color)
        f.pill_label.configure(bg=color)

    def _refresh_telemetry(self):
        with self.tele_lock:
            s = dict(self.tele_state)

        def age(k):
            t = s.get(k + "_t")
            return None if t is None else time.time() - t

        px4_ok = age("nav_state") is not None and age("nav_state") < 2.0
        self._set_pill("px4", px4_ok)

        cam_age = age("frame")
        self._set_pill("camera", cam_age is not None and cam_age < 2.0)

        marker_age = age("markers")
        self._set_pill("aruco", marker_age is not None and marker_age < 2.0)

        rpy = s.get("rpy")
        rpy_age = age("rpy")
        if rpy is not None and rpy_age is not None and rpy_age < 1.0:
            roll, pitch, yaw = rpy
            self.att_label.configure(
                text=f"Roll {roll:+6.1f}   Pitch {pitch:+6.1f}   Yaw {yaw:+6.1f}",
                fg="#1e7e34")
            self.att_age_label.configure(text=f"수신 {rpy_age * 1000:4.0f}ms 전")
            quat = s.get("quat")
            if quat is not None:
                self._draw_attitude_3d(*quat)
        else:
            self.att_label.configure(text="Roll ─   Pitch ─   Yaw ─", fg="#999999")
            # 위 "수신 Nms 전"이랑 길이 비슷하게 맞춰서(짧은 문구) 라벨 폭이 안 흔들리게 함
            self.att_age_label.configure(text="수신 끊김" if rpy is not None else "")

        batt = s.get("battery")
        batt_age = age("battery")
        if batt is not None and batt_age is not None and batt_age < 2.0:
            voltage, remaining, warning = batt
            pct = f"{remaining * 100:.0f}%" if remaining is not None and remaining >= 0 else "?%"
            # PX4 BatteryStatus.warning: 0=정상 1=LOW 2=CRITICAL 3=EMERGENCY 4=FAILED
            color = "#1e7e34" if warning == 0 else ("#e65100" if warning == 1 else "#c62828")
            self.batt_label.configure(text=f"배터리 {voltage:4.1f}V {pct:>4}", fg=color)
        else:
            self.batt_label.configure(text="배터리 ─", fg="#999999")

        flight = s.get("flight")
        flight_age = age("flight")
        if flight is not None and flight_age is not None and flight_age < 1.0:
            vx, vy, vz, rel_alt = flight
            hspeed = math.hypot(vx, vy)
            vspeed = -vz  # NED: 아래가 +라 부호 뒤집어야 "상승=+"로 직관적
            self.flight_relalt_label.configure(
                text=f"상대고도: {rel_alt:5.2f} m" if rel_alt is not None else "상대고도: ─",
                fg="#1e7e34")
            self.flight_hspeed_label.configure(text=f"수평속도: {hspeed:5.2f} m/s", fg="#1e7e34")
            self.flight_vspeed_label.configure(text=f"수직속도: {vspeed:+5.2f} m/s", fg="#1e7e34")
        else:
            self.flight_relalt_label.configure(text="상대고도: ─", fg="#999999")
            self.flight_hspeed_label.configure(text="수평속도: ─", fg="#999999")
            self.flight_vspeed_label.configure(text="수직속도: ─", fg="#999999")

        mission_state = s.get("mission_state")
        mission_age = age("mission_state")
        if mission_state is not None and mission_age is not None and mission_age < 2.0:
            self.mission_state_label.configure(text=f"미션: {mission_state}", fg="#1e7e34")
        else:
            self.mission_state_label.configure(text="미션: ─", fg="#999999")

        armed = s.get("armed")
        armed_age = age("armed")
        if armed is not None and armed_age is not None and armed_age < 2.0:
            # armed 상태는 "위험 신호"라 빨강, disarmed는 안전하니 초록 - 다른
            # 값들(초록=데이터 정상수신)과 색 의미가 반대라서 헷갈리지 않게 주의.
            self.arm_state_label.configure(
                text="ARM: ARMED" if armed else "ARM: DISARMED",
                fg="#c62828" if armed else "#1e7e34")
        else:
            self.arm_state_label.configure(text="ARM: ─", fg="#999999")

        safety = s.get("safety")
        safety_age = age("safety")
        if safety is not None and safety_age is not None and safety_age < 2.0:
            local_inv, global_inv, home_inv = safety
            # GPS 없는 비전(ArUco) 기반 세팅에서는 global(위경도) 위치는
            # 설계상 절대 못 얻는다 - 그런데도 global_inv를 같이 보면
            # local이 멀쩡해도 영원히 "불가"로 잘못 뜸(2026-09-21 실측:
            # EKF2_EV_CTRL로 융합 성공해서 local xy_valid/z_valid 다
            # True인데도 GUI만 계속 불가로 표시됨). OFFBOARD 로컬제어엔
            # global 필요 없으니 local만 본다.
            if local_inv:
                self.safety_label.configure(text="위치추정: 불가", fg="#c62828")
            elif home_inv:
                self.safety_label.configure(text="위치추정: OK(홈 없음)", fg="#e65100")
            else:
                self.safety_label.configure(text="위치추정: OK", fg="#1e7e34")
        else:
            self.safety_label.configure(text="위치추정: ─", fg="#999999")

        # 이 줄이 예전에 _draw_attitude_3d() 끝으로 잘못 들어가있던 버그가 있었음
        # (2026-09-08 실측) - quat이 아직 None인 동안(자세 데이터 도착 전)은
        # _draw_attitude_3d()가 아예 호출이 안 되니 재스케줄 자체가 안 일어나서,
        # 첫 틱에 데이터가 없으면 이 갱신 루프가 그 자리에서 영구히 멈춰버렸음
        # (나중에 데이터가 들어와도 다시는 안 살아남).
        self.root.after(100, self._refresh_telemetry)

    def _draw_attitude_3d(self, w, x, y, z):
        R = _quat_to_rotmat(w, x, y, z)
        segs = [[_ned_to_enu_display(R @ np.array(p0)), _ned_to_enu_display(R @ np.array(p1))]
                for p0, p1 in _ATT_BODY_LINES[:2]]
        self.att_lc.set_segments(segs)
        nose_segs = [[_ned_to_enu_display(R @ np.array(p0)), _ned_to_enu_display(R @ np.array(p1))]
                     for p0, p1 in _ATT_BODY_LINES[2:]]
        self.att_nose_lc.set_segments(nose_segs)
        pts = np.array([_ned_to_enu_display(R @ p) for p in _ATT_MOTOR_POINTS_BODY])
        self.att_motor_scatter._offsets3d = (pts[:, 0], pts[:, 1], pts[:, 2])
        self.att_canvas.draw_idle()

    @staticmethod
    def _widest_run(row):
        """1D 이진 배열(row)에서 "라인일 법한 폭"의 연속 non-zero 구간 중
        가장 넓은 것의 (중심 x, 폭) 반환. line_tracker.py의 _row_centroid와
        완전히 동일한 기준(폭 상한 포함)이라 여기서 계산한 폭/위치가 실제
        라인트레이서 동작과 일치한다. 상한을 넘는 구간(비네팅/그림자 등)
        만 있으면 None(못 찾음)을 반환한다 - 엉뚱한 위치를 자신있게
        표시하는 것보다 안전함."""
        xs = np.nonzero(row)[0]
        if xs.size == 0:
            return None
        gaps = np.where(np.diff(xs) > 1)[0]
        starts = np.concatenate(([0], gaps + 1))
        ends = np.concatenate((gaps, [len(xs) - 1]))
        widths = xs[ends] - xs[starts] + 1

        max_reasonable_width = max(20, int(0.15 * len(row)))
        valid = widths <= max_reasonable_width
        if not np.any(valid):
            return None

        valid_idx = np.where(valid)[0]
        best = valid_idx[np.argmax(widths[valid_idx])]
        run = xs[starts[best]:ends[best] + 1]
        return float(run.mean()), int(run[-1] - run[0] + 1)

    def _process_linetrack(self, arr_rgb):
        """arr_rgb: RGB numpy 배열. 기존 my_first_pkg/line_tracker.py의
        LineTracker를 그대로 재사용(로직 복제 안 함) — search_phase_node.py의
        그리드 라인 추종에 실제로 쓰이는 것과 동일한 검출 결과를 여기서
        그대로 확인할 수 있게 오버레이만 추가."""
        bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
        result = self.line_tracker.process(bgr)

        h, w = arr_rgb.shape[:2]
        y0 = int(h * self.line_tracker.roi_top_ratio)
        y1 = int(h * self.line_tracker.roi_bottom_ratio)
        cv2.rectangle(arr_rgb, (0, y0), (w - 1, y1), (255, 193, 7), 2)  # ROI (노랑)
        cv2.line(arr_rgb, (w // 2, y0), (w // 2, y1), (160, 160, 160), 1)  # 화면 중앙 기준선

        if result.line_found:
            bottom_x = int(w / 2 + result.e_y)
            bottom_x = max(0, min(w - 1, bottom_x))
            deg = math.degrees(result.e_psi)

            # ---- 상단(top_c) 지점 표시 + 품질 점수(0~100) ----
            # top_c는 LineTracker.process()가 방금 내부에서 실제로 쓴 값을
            # 그대로 가져다 쓴다(_last_top_x) - 따로 재계산하면 연속성
            # 보정(발 등 방해물 대응)이 반영 안 된 값이 나와 e_psi와 안
            # 맞을 수 있어서, 여기서는 "진짜 쓰인 값"만 그린다.
            roi_bgr = bgr[y0:y1, :]
            gray_roi = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray_roi, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            if cv2.countNonZero(mask) > mask.size // 2:
                mask = cv2.bitwise_not(mask)
            bottom_row = mask[-5, :] if mask.shape[0] > 5 else mask[-1, :]
            bottom_run = self._widest_run(bottom_row)
            width_ratio = (bottom_run[1] / w) if bottom_run else 0.0

            top_y_full = y0 + (5 if mask.shape[0] > 5 else 0)
            if self.line_tracker._last_top_x is not None:
                top_x = int(round(self.line_tracker._last_top_x))
                cv2.line(arr_rgb, (bottom_x, y1 - 5), (top_x, top_y_full), (52, 168, 83), 1)
                cv2.circle(arr_rgb, (top_x, top_y_full), 6, (217, 42, 53), -1)  # 상단=빨강

            cv2.circle(arr_rgb, (bottom_x, y1 - 5), 6, (52, 168, 83), -1)  # 하단=초록 (위에 겹쳐 그림)

            # 두꺼운 라인일수록 모션 블러/흔들림에 더 잘 버틴다는 게 실측
            # 확인됐음(2026-09-07: 얇은 라인은 블러 커지면 금방 놓치지만
            # 두꺼운 라인은 훨씬 오래 버팀) -> "두꺼울수록 감점"이 아니라
            # "두꺼울수록(신뢰 상한선 안에서) 가점"으로 바꿈. 상한(15%)을
            # 넘는 폭은 애초에 _widest_run에서 이미 걸러져서 못 찾음으로
            # 처리되므로, 여기 도달한 값은 항상 "라인일 법한 폭"이다.
            score = 70
            if width_ratio < 0.01:
                score -= 40   # 폭이 거의 없음 -> 노이즈성 검출일 가능성
            else:
                score += min(30, int(width_ratio / 0.15 * 30))  # 두꺼울수록 최대 +30
            if abs(deg) > 45:
                score -= 25   # 각도가 과도함 -> 라인이 아닌 걸 잡았을 가능성
            self._line_streak = getattr(self, "_line_streak", 0) + 1
            score += min(15, self._line_streak)  # 연속 검출 안정성 보너스
            score = max(0, min(100, score))

            cross = " CROSS" if result.intersection else ""
            status = f"LINE e_y={result.e_y:+.0f}px e_psi={deg:+.1f}deg score={score}/100{cross}"
        else:
            self._line_streak = 0
            status = "LINE: not found"

        return arr_rgb, result.line_found, status

    def _detect_boundary(self, arr):
        """arr: RGB numpy 배열. 착륙패드 경계로 추정되는 사각형 윤곽을 찾아
        초록색으로 그려서 반환. (Canny + contour 기반, 색상 가정 없음)

        기존 버전은 화면에 보이는 아무 사각형(모니터 등)이나 다 잡는 오탐이
        심했음 -> 추가 필터(경계 접촉 제외/종횡비/볼록성)와 프레임 간 연속
        검출 요구(히스테리시스)로 안정성을 높임."""
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # 고정 임계값 대신 이미지 밝기 분포 기반 자동 임계값 (조명 변화에 더 강함)
        med = float(np.median(blur))
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * med))
        upper = int(min(255, (1.0 + sigma) * med))
        edges = cv2.Canny(blur, lower, upper)
        edges = cv2.dilate(edges, None, iterations=1)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        h, w = gray.shape
        frame_area = h * w
        try:
            min_area_pct = float(self.boundary_min_area_var.get()) / 100.0
        except (ValueError, AttributeError):
            min_area_pct = 0.015
        margin = 4

        best, best_area = None, 0
        for cnt in contours:
            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)
            if hull_area <= 0:
                continue
            solidity = cv2.contourArea(cnt) / hull_area
            if solidity < 0.9:
                continue  # 뭉개지거나 노치 있는 형태는 제외

            peri = cv2.arcLength(hull, True)
            approx = cv2.approxPolyDP(hull, 0.02 * peri, True)
            if len(approx) != 4 or not cv2.isContourConvex(approx):
                continue

            xs, ys = approx[:, 0, 0], approx[:, 0, 1]
            if xs.min() <= margin or ys.min() <= margin or xs.max() >= w - margin or ys.max() >= h - margin:
                continue  # 화면 경계에 닿아있으면 창문/모니터 등일 가능성 높음 -> 제외

            bx, by, bw, bh = cv2.boundingRect(approx)
            aspect = min(bw, bh) / max(bw, bh)
            if aspect < 0.6:
                continue  # 너무 길쭉함 (예: 16:9 모니터/창문)

            area = cv2.contourArea(approx)
            if area > min_area_pct * frame_area and area < 0.6 * frame_area and area > best_area:
                best, best_area = approx, area

        # 프레임 하나 튄다고 바로 깜빡이지 않도록 연속 검출/미검출 요구
        if best is not None:
            self._boundary_hit_streak = getattr(self, "_boundary_hit_streak", 0) + 1
            self._boundary_miss_streak = 0
            self._boundary_last = best
            self._boundary_last_area = best_area
        else:
            self._boundary_miss_streak = getattr(self, "_boundary_miss_streak", 0) + 1
            if self._boundary_miss_streak >= 5:
                self._boundary_hit_streak = 0
                self._boundary_last = None

        stable = getattr(self, "_boundary_hit_streak", 0) >= 3 and getattr(self, "_boundary_last", None) is not None
        if stable:
            best = self._boundary_last
            cv2.polylines(arr, [best], True, (52, 168, 83), 3)
            m = cv2.moments(best)
            if m["m00"] != 0:
                cx, cy = int(m["m10"] / m["m00"]), int(m["m01"] / m["m00"])
                cv2.drawMarker(arr, (cx, cy), (52, 168, 83), cv2.MARKER_CROSS, 16, 2)

            # 라인트레이서와 같은 방향: 화면에서 더 크게(뚜렷하게) 잡힐수록
            # 흔들림/블러에도 잘 버티고 오탐 가능성도 낮으므로 더 높은 점수.
            area_ratio = self._boundary_last_area / frame_area
            score = 60 + min(30, int((area_ratio - min_area_pct) / (0.6 - min_area_pct) * 30))
            score += min(10, self._boundary_hit_streak)
            score = max(0, min(100, score))
            return arr, True, f"BOUNDARY DETECTED score={score}/100"
        return arr, False, "BOUNDARY: searching..."

    def _process_calibration(self, arr_full):
        """arr_full: 원본 해상도 RGB numpy 배열. in-place로 체스보드 코너를
        그리고, 찾은 코너는 self.calib_last_* 에 저장해서 '캡처' 버튼이
        실제 캘리브레이션 샘플로 채택할 수 있게 한다."""
        gray = cv2.cvtColor(arr_full, cv2.COLOR_RGB2GRAY)
        try:
            cols = int(self.calib_cols_var.get())
            rows = int(self.calib_rows_var.get())
        except ValueError:
            cols, rows = 9, 6
        pattern_size = (cols, rows)

        found_cb, corners = cv2.findChessboardCorners(
            gray, pattern_size,
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)

        n = len(self.calib_imgpoints)
        if found_cb:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            cv2.drawChessboardCorners(arr_full, pattern_size, corners, found_cb)
            self.calib_last_corners = corners
            self.calib_last_pattern = pattern_size
            self.calib_last_image_size = (gray.shape[1], gray.shape[0])
            status = f"CHESSBOARD FOUND (samples: {n})"
        else:
            self.calib_last_corners = None
            status = f"searching chessboard... (samples: {n})"
        return found_cb, status

    def _calib_capture(self):
        if self.calib_last_corners is None:
            self._append_log("[calib] 체스보드가 화면에 안 잡혀서 캡처 못 함\n")
            return
        cols, rows = self.calib_last_pattern
        try:
            square_mm = float(self.calib_square_mm_var.get())
        except ValueError:
            square_mm = 25.0
        objp = np.zeros((cols * rows, 3), np.float32)
        objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
        self.calib_objpoints.append(objp)
        self.calib_imgpoints.append(self.calib_last_corners)
        self.calib_sample_label.configure(text=f"샘플: {len(self.calib_imgpoints)}")
        self._append_log(f"[calib] 샘플 캡처됨 (총 {len(self.calib_imgpoints)}개)\n")

    def _calib_reset(self):
        self.calib_objpoints = []
        self.calib_imgpoints = []
        self.calib_last_corners = None
        self.calib_sample_label.configure(text="샘플: 0")
        self._append_log("[calib] 샘플 초기화됨\n")

    def _calib_compute(self):
        n = len(self.calib_imgpoints)
        if n < 8:
            messagebox.showwarning("샘플 부족", f"최소 8장 이상 필요합니다 (현재 {n}장).\n"
                                    "체스보드를 여러 각도/거리로 비춰서 '캡처'를 더 눌러주세요.")
            return
        if self.calib_last_image_size is None:
            messagebox.showerror("오류", "이미지 크기 정보가 없습니다.")
            return

        self._append_log(f"[calib] {n}장으로 캘리브레이션 계산 중...\n")
        ret, K, D, rvecs, tvecs = cv2.calibrateCamera(
            self.calib_objpoints, self.calib_imgpoints, self.calib_last_image_size, None, None)

        w, h = self.calib_last_image_size
        yaml_text = (
            f"image_width: {w}\n"
            f"image_height: {h}\n"
            f"camera_name: default_cam\n"
            f"camera_matrix:\n"
            f"  rows: 3\n  cols: 3\n"
            f"  data: [{K[0,0]}, {K[0,1]}, {K[0,2]}, {K[1,0]}, {K[1,1]}, {K[1,2]}, {K[2,0]}, {K[2,1]}, {K[2,2]}]\n"
            f"distortion_model: plumb_bob\n"
            f"distortion_coefficients:\n"
            f"  rows: 1\n  cols: 5\n"
            f"  data: [{D[0,0]}, {D[0,1]}, {D[0,2]}, {D[0,3]}, {D[0,4]}]\n"
            f"rectification_matrix:\n"
            f"  rows: 3\n  cols: 3\n"
            f"  data: [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]\n"
            f"projection_matrix:\n"
            f"  rows: 3\n  cols: 4\n"
            f"  data: [{K[0,0]}, 0.0, {K[0,2]}, 0.0, 0.0, {K[1,1]}, {K[1,2]}, 0.0, 0.0, 0.0, 1.0, 0.0]\n"
        )
        with open(self.CALIB_SAVE_PATH, "w") as f:
            f.write(yaml_text)

        msg = f"캘리브레이션 완료. 재투영 오차(RMS): {ret:.4f}px\n저장: {self.CALIB_SAVE_PATH}"
        self._append_log(f"[calib] {msg}\n")
        messagebox.showinfo(
            "캘리브레이션 완료",
            msg + "\n\ncamera 서브시스템을 재시작하면 이 보정값이 자동 적용됩니다.",
        )

    @staticmethod
    def _quat_to_rvec(x, y, z, w):
        n = (x * x + y * y + z * z + w * w) ** 0.5
        if n < 1e-9:
            return np.zeros((3, 1))
        x, y, z, w = x / n, y / n, z / n, w / n
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])
        rvec, _ = cv2.Rodrigues(R)
        return rvec

    def _draw_marker_axes(self, arr_full):
        """arr_full: 원본 해상도 RGB numpy 배열. 감지된 각 마커에 대해
        aruco_tracker가 이미 계산해서 /aruco_detections로 보내주는 pose를
        그대로 이용해 XYZ 축을 투영해서 그린다 (별도 pose 재계산 없음)."""
        with self.tele_lock:
            K = self.tele_state.get("cam_K")
            D = self.tele_state.get("cam_D")
            poses = self.tele_state.get("marker_poses", [])
        if K is None or not poses:
            return

        axis_len = 0.075  # 15cm 마커 기준 절반 길이(m) - 보기 좋은 정도
        axis_pts = np.float32([
            [0, 0, 0], [axis_len, 0, 0], [0, axis_len, 0], [0, 0, axis_len],
        ]).reshape(-1, 3)

        for mid, px, py, pz, qx, qy, qz, qw in poses:
            rvec = self._quat_to_rvec(qx, qy, qz, qw)
            tvec = np.array([px, py, pz], dtype=np.float64)
            try:
                img_pts, _ = cv2.projectPoints(axis_pts, rvec, tvec, K, D)
            except cv2.error:
                continue
            img_pts = img_pts.reshape(-1, 2).astype(int)
            origin, x_pt, y_pt, z_pt = [tuple(p) for p in img_pts]
            cv2.line(arr_full, origin, x_pt, (255, 0, 0), 3)     # X = 빨강
            cv2.line(arr_full, origin, y_pt, (0, 200, 0), 3)     # Y = 초록
            cv2.line(arr_full, origin, z_pt, (40, 110, 255), 3)  # Z = 파랑
            cv2.circle(arr_full, origin, 5, (255, 255, 255), -1)

    def _refresh_camera(self):
        with self.tele_lock:
            frame = self.tele_state.get("frame")
            frame_age = time.time() - self.tele_state.get("frame_t", 0) if "frame_t" in self.tele_state else None
            marker_info = self.tele_state.get("marker_info", [])
            marker_age = time.time() - self.tele_state.get("markers_t", 0) if "markers_t" in self.tele_state else None

        if frame is not None and frame_age is not None and frame_age < 2.0:
            # cam_box는 pack_propagate(False)로 고정되어 있어 라벨/이미지 크기가
            # 박스 크기에 영향을 못 준다 -> cam_box 크기를 기준으로만 계산하면
            # (라벨 자기 크기를 기준 삼을 때 생기는) 피드백 루프가 생기지 않는다.
            box_w = self.cam_box.winfo_width() or self.CAM_DISPLAY_SIZE[0]
            box_h = self.cam_box.winfo_height() or self.CAM_DISPLAY_SIZE[1]
            target = (max(box_w - 16, 200), max(box_h - 16, 120))

            mode = self.cam_mode.get()

            if mode == "calib":
                # 코너 검출 정확도를 위해 축소 전 원본 해상도에서 처리하고,
                # 검출 결과가 그려진 이미지를 그 다음에 화면 크기로 축소한다.
                arr_full = frame.copy()
                found, status = self._process_calibration(arr_full)
                img = PILImage.fromarray(arr_full).convert("RGB")
                img.thumbnail(target)
                border_color = "#34a853" if found else "#d93025"
            else:
                # XYZ 축을 원근 왜곡 없이 정확히 그리려면 카메라 내부파라미터가
                # 정의된 원본 해상도에서 투영해야 하므로 축소 전에 그린다.
                # boundary/linetrack 모드도 축소 전 프레임에 먼저 축을 그려두고,
                # 그 다음에 (기존과 동일하게) 축소된 이미지 위에서 각자 검출을 수행한다.
                arr_full = frame.copy()
                aruco_found = marker_age is not None and marker_age < 2.0 and marker_info
                if aruco_found:
                    parts = [f"ID {mid} ({dist:.2f}m)" for mid, dist in marker_info]
                    aruco_status = "ArUco DETECTED: " + ", ".join(parts)
                    self._draw_marker_axes(arr_full)
                else:
                    aruco_status = "ArUco: searching..."

                if mode == "normal":
                    border_color = "#34a853" if aruco_found else "#d93025"
                    status = aruco_status

                img = PILImage.fromarray(arr_full).convert("RGB")
                img.thumbnail(target)

            if mode == "boundary":
                arr = np.array(img)  # 화면 크기로 이미 축소된 RGB 배열 (성능상 여기서 검출)
                arr, found, status = self._detect_boundary(arr)
                img = PILImage.fromarray(arr)
                border_color = "#34a853" if found else "#d93025"
            elif mode == "linetrack":
                arr = np.array(img)
                arr, found, status = self._process_linetrack(arr)
                img = PILImage.fromarray(arr)
                border_color = "#34a853" if found else "#d93025"

            draw = ImageDraw.Draw(img)
            font_size = max(14, img.width // 32)
            try:
                font = ImageFont.truetype(
                    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
            except Exception:
                font = ImageFont.load_default()

            # DejaVuSans에 한글 글리프가 없어 깨지므로 오버레이는 영어로 표기
            bw = 4
            draw.rectangle([0, 0, img.width - 1, img.height - 1], outline=border_color, width=bw)
            text_bbox = draw.textbbox((0, 0), status, font=font)
            tw, th = text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1]
            draw.rectangle([bw, bw, bw + tw + 16, bw + th + 14], fill=border_color)
            draw.text((bw + 8, bw + 6), status, fill="white", font=font)

            if mode in ("boundary", "linetrack"):
                # 상단 배너는 해당 모드 자체(경계/라인) 검출 상태용이라
                # ArUco 상태는 하단에 별도 배너로 함께 표시한다.
                aruco_color = "#34a853" if aruco_found else "#5f6368"
                a_bbox = draw.textbbox((0, 0), aruco_status, font=font)
                atw, ath = a_bbox[2] - a_bbox[0], a_bbox[3] - a_bbox[1]
                ay1 = img.height - bw - ath - 14
                draw.rectangle([bw, ay1, bw + atw + 16, img.height - bw], fill=aruco_color)
                draw.text((bw + 8, ay1 + 6), aruco_status, fill="white", font=font)

            if self.flip_var.get():
                # 오버레이(축/박스/텍스트)까지 다 그려진 최종 이미지를 통째로
                # 뒤집는다 - 검출 좌표는 원본 프레임 기준이라 미리 뒤집으면
                # 오버레이 위치가 틀어짐(원근 투영/라인트레이서 다 원본 좌표계
                # 기준으로 계산됨).
                img = img.transpose(PILImage.FLIP_LEFT_RIGHT)

            photo = ImageTk.PhotoImage(img)
            self.cam_label.configure(image=photo, text="")
            self.cam_label.image = photo
        elif frame is None:
            self.cam_label.configure(image="", text="cam\ncamera를 켜세요")
            self.cam_label.image = None

        self.root.after(50, self._refresh_camera)

    def _on_close(self):
        for sub in self.subsystems.values():
            sub.stop()
        self.direct_cam.stop()
        try:
            rclpy.shutdown()
        except Exception:
            pass
        try:
            self.log_file.close()
        except Exception:
            pass
        self.root.destroy()


# ---- 창 2개 이상 동시 실행 방지 + 이전 세션 고아 프로세스 정리 ----
# (2026-09-21 실측: 창을 닫지 않고 새로 하나 더 띄우면 각자 자기
# MicroXRCEAgent를 같은 /dev/ttyTHS0에 서로 다른 baud로 띄워서 서로
# 충돌 -> 자세각(UART 점검)이 영영 안 들어옴. 창을 하나만 띄우게
# 강제하고, kill -9/크래시로 죽은 이전 세션이 남긴 서브시스템도 자동
# 정리해서 같은 문제가 재발 안 하게 함.)
_INSTANCE_LOCK_PATH = "/tmp/kasa_main_panel.lock"

_ORPHAN_KILL_PATTERNS = [
    "MicroXRCEAgent",
    "usb_cam_node_exe",
    "aruco_tracker_autostart",
    "precision_landing landing",
    "search_phase_node.py",
]


def _cleanup_orphan_subsystems():
    for pattern in _ORPHAN_KILL_PATTERNS:
        subprocess.run(["pkill", "-f", pattern], capture_output=True)


def _ensure_single_instance():
    # flock은 파일이 아니라 프로세스/fd에 묶여있어서, 이전 프로세스가
    # kill -9로 죽어도 OS가 자동으로 락을 풀어준다(pid 파일 존재 여부만
    # 보는 방식과 달리 "죽었는데 락은 안 풀림" 문제가 없음).
    lock_fp = open(_INSTANCE_LOCK_PATH, "w")
    try:
        fcntl.flock(lock_fp, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        root = tk.Tk()
        root.withdraw()
        messagebox.showerror(
            "KASA 정밀착륙 제어판",
            "이미 다른 창이 실행 중입니다.\n"
            "기존 창을 사용하거나 닫은 뒤 다시 실행하세요.\n"
            "(두 개를 같이 띄우면 xrce 등 서브시스템이 같은 포트를 두고 서로 충돌합니다.)",
        )
        root.destroy()
        # exit(1)을 쓰면 run_main.sh가 이걸 크래시로 오인해서 계속
        # 재시작을 시도 -> 이 에러 팝업이 2초마다 무한 반복되는 사고가
        # 있었음(2026-09-21 실측). "이미 실행 중"은 정상적인 상황이므로
        # exit(0)으로 run_main.sh의 재시작 루프가 그냥 끝나게 한다.
        sys.exit(0)
    lock_fp.write(str(os.getpid()))
    lock_fp.flush()
    return lock_fp  # 참조를 계속 들고 있어야 락이 안 풀림(GC로 fd 닫히는 것 방지)


if __name__ == "__main__":
    _lock_fp = _ensure_single_instance()
    _cleanup_orphan_subsystems()
    root = tk.Tk()
    App(root)
    root.mainloop()

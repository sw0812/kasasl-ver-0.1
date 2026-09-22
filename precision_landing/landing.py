#!/usr/bin/env python3
"""
ArUco 마커 기반 PX4 정밀 착륙 노드.

통신 경로: ROS2 <-> MicroXRCEAgent <-> PX4 uxrce_dds_client (MAVROS 아님)
필요 조건:
  - px4_msgs 패키지가 이 워크스페이스에 빌드되어 있어야 함 (PX4 버전과 msg 정의가 맞아야 함)
  - MicroXRCEAgent udp4 -p 8888 가 떠 있어야 함
  - PX4가 이 포트로 uxrce_dds_client를 이미 실행 중이어야 함 (부팅 로그에서 확인됨)

좌표계 가정 (반드시 실제 환경에 맞춰 검증할 것):
  - 카메라는 완전 하향(pitch 90도)으로 장착되어 있다고 가정 (기체에 고정, 짐벌 없음)
  - ArUco 카메라 좌표계: x=오른쪽, y=아래, z=카메라 정면(마커까지의 거리)
  - 이를 body frame(x=전방, y=우측)으로 매핑: body_x = -camera_y, body_y = camera_x
    (마커가 카메라 화면에서 위(-y)에 있으면 드론이 전방으로 가야 마커에 접근)
  - yaw는 최초 진입 시 heading으로 한 번 고정하고 이후 바꾸지 않는다. 이전엔 마커 쪽으로
    기체를 계속 회전시키며(pursuit) 그 방향으로 전진하는 방식을 썼는데, 목표 근처에서는
    위치가 조금만 흔들려도 bearing이 급격히 튀어서 MAX_YAW_RATE를 못 따라가 목표 주변을
    도는(orbit) 현상이 생겼다(실측 1.38m 부근). 그래서 yaw와 이동 방향을 분리해, body
    frame 오프셋을 현재 heading으로 회전시켜 바로 world-frame vx/vy로 쓴다 — 기체가
    어느 쪽을 보고 있든 마커 쪽으로 옆이동까지 포함해 직선으로 수렴한다.

터미널 표시:
  - 매 tick마다 로그를 새 줄로 찍으면 터미널이 도배되므로, 상태(state/마커/고도/오차/
    속도/yaw)를 한 줄에 모아 \r(캐리지리턴)로 계속 덮어쓴다 (status_bar()).
  - 실제 갱신은 STATUS_REFRESH_HZ로 스로틀링해서 눈으로 읽을 수 있는 속도로만 찍는다.
  - 마커 확인/유실 등 의미 있는 이벤트는 같은 줄에 메시지로 얹어서 보여준다.

OFFBOARD/ARM 진입:
  - "명령을 한 번 보냈다"와 "실제로 그 상태가 됐다"는 다르다. DO_SET_MODE나 ARM 명령이
    타이밍(예: 이미 비행 중에 스크립트를 재시작하는 경우) 때문에 거부될 수 있는데,
    예전엔 딱 한 번만 보내고 끝이라 거부되면 그대로 원래 모드(Hold/Position 등)에
    머물러버렸다 -> 속도 명령을 아무리 보내도 무시되고 제자리 고도만 유지되는 버그.
    그래서 VehicleStatus로 실제 nav_state/arming_state가 OFFBOARD+ARMED로 확인될
    때까지 주기적으로 계속 재시도한다(확인되면 재시도 중단).
  - 단, NAV_LAND를 보낸 뒤에는 얘기가 다르다: PX4가 스스로 AUTO.LAND 모드로 전환하는데,
    이걸 계속 OFFBOARD로 재시도해서 뺏어오면 PX4의 자체 착륙 로직과 충돌한다. 그래서
    land_cmd_sent 이후에는 OFFBOARD 재시도를 멈춘다.

LAND 전환 타임아웃:
  - 고도 조건(LAND_ALTITUDE 이하)과 정렬 조건(err < CENTER_TOLERANCE)을 동시에 만족해야
    LAND로 넘어가는데, 낮은 고도일수록 오차가 흔들리기 쉬워서 두 조건이 동시에 안 맞아
    APPROACH<->DESCEND만 왔다갔다 하며 영원히 호버링하는 경우가 있었다. 그래서 고도
    조건을 만족한 채로 LAND_STUCK_TIMEOUT_SEC 이상 지나면, 정렬이 완벽하지 않아도
    안전하게 착륙을 강행한다(공중에서 무한 대기하는 것보다 낫다).

    [수정 메모] 이 stuck 타이머는 예전엔 _land_stuck_since를 APPROACH 진입마다
    None으로 리셋해버려서 실제로는 절대 누적되지 않는 죽은 코드였다. 지금은
    "현재 고도 < LAND_ALTITUDE 인가"만 보고 상태(APPROACH/DESCEND) 전환과
    무관하게 독립적으로 누적하도록 고쳤다. 그래야 err가 CENTER_TOLERANCE
    경계를 넘나들며 APPROACH<->DESCEND를 왕복해도 타이머가 끊기지 않는다.
"""

import math
import traceback
import sys
import shutil
import re

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from aruco_opencv_msgs.msg import ArucoDetection
from px4_msgs.msg import (
    OffboardControlMode,
    TrajectorySetpoint,
    VehicleCommand,
    VehicleStatus,
    VehicleLocalPosition,
)


# PX4 uXRCE-DDS 기본 QoS: BEST_EFFORT + VOLATILE + KEEP_LAST
# 이걸 안 맞추면 subscribe/publish가 에러 없이 그냥 안 붙는다.
PX4_QOS = QoSProfile(
    reliability=ReliabilityPolicy.BEST_EFFORT,
    durability=DurabilityPolicy.VOLATILE,
    history=HistoryPolicy.KEEP_LAST,
    depth=5,
)

# 상태 머신
STATE_SEARCH = "SEARCH"
STATE_APPROACH = "APPROACH"
STATE_DESCEND = "DESCEND"
STATE_LAND = "LAND"

# 튜닝 파라미터
# [2026-09-08 수정] 원래 1.2였음 - search_phase_node.py에 이 로직을 이식한 뒤
# 실비행에서 ArUco 검출이 뜨문뜨문(sparse) + EMA 필터 지연과 겹쳐 이 게인이면
# 수렴 안 하고 목표 주변을 계속 도는 한계궤도(limit cycle)가 실측 확인됨 - 게인을
# 낮춰서 지연 대비 과도한 반응을 줄임. 그 수정을 뒤늦게 이 원본 파일에도 반영.
KP_XY = 0.5               # 수평 위치 오차 -> 속도 게인
MAX_XY_SPEED = 0.5        # m/s, 수평 속도 상한
MAX_XY_ACCEL = 1.6        # m/s^2, 속도 변화율 제한
DESCEND_SPEED = 0.1       # m/s, 하강 속도 (지면 근접 안전 때문에 그대로 유지)
CENTER_TOLERANCE = 0.15   # m, 이 안에 들어오면 "정렬됨"으로 판단
LAND_ALTITUDE = 0.48       # m, 이 고도 이하에서 최종 저속 하강 시작
LAND_FINAL_SPEED = 0.12   # m/s, LAND 상태에서의 하강 속도 (DESCEND_SPEED보다 느리게, 지면 근접 안전용)
DISARM_ALTITUDE = 0.05    # m, 이 고도 이하로 내려오면 강제 disarm
LAND_STUCK_TIMEOUT_SEC = 3.0  # 초, 고도 조건은 맞는데 정렬이 안 맞아 착륙을 못 하고
                               # 있는 상태가 이만큼 지속되면 정렬 무시하고 착륙 강행
MARKER_LOST_TIMEOUT = 2.7  # 초, 이 시간 넘게 마커 안 보이면 정지/유지
TIMER_PERIOD = 0.02        # 초, 아래 create_timer 주기와 반드시 일치시킬 것 (50Hz)

MARKER_FILTER_ALPHA = 0.3  # 마커 좌표 EMA 필터 계수 (0~1, 작을수록 더 부드럽지만 반응 느려짐)

# OFFBOARD/ARM 확인 전까지 재시도하는 주기 (너무 자주 보내면 커맨드 버스에 부담)
MODE_RETRY_PERIOD_SEC = 0.5
MODE_RETRY_EVERY_N_TICKS = max(1, round(MODE_RETRY_PERIOD_SEC / TIMER_PERIOD))

# 터미널 상태줄 갱신 주기 (너무 빠르면 눈으로 못 읽고 깜빡여서 보임)
STATUS_REFRESH_HZ = 10
STATUS_REFRESH_EVERY_N_TICKS = max(1, round((1.0 / TIMER_PERIOD) / STATUS_REFRESH_HZ))

# ANSI 컬러
COLOR_GREEN = "\033[92m"
COLOR_RED = "\033[91m"
COLOR_RESET = "\033[0m"

# status_bar()에서 터미널 폭 계산 시 색상 코드는 빼고 세기 위한 정규식
_ANSI_RE = re.compile(r"\033\[[0-9;]*m")


class PrecisionLanding(Node):

    def __init__(self):
        super().__init__("precision_landing")

        # 마커 관련 상태
        self.marker_visible = False
        self.marker_confirmed = False  # 타임아웃 기준 "지금 마커 잡혀있음" 상태 (로그 토글용)
        self.marker_ever_seen = False  # 실제로 한 번이라도 마커를 본 적 있는지 (초기 오탐 FOUND 방지)
        self.last_marker_time = self.get_clock().now()
        self.marker_x = 0.0  # camera frame
        self.marker_y = 0.0
        self.marker_z = 0.0  # 마커까지의 거리(대략적인 고도 추정에도 참고 가능)

        # 속도 명령 (NED)
        self.vx = 0.0
        self.vy = 0.0
        self.vz = 0.0

        # PX4 상태
        self.arming_state = None
        self.nav_state = None
        self.current_altitude = 0.0  # NED z, 아래가 양수이므로 고도 = -z
        self.current_heading = 0.0   # PX4 heading(rad), local_position_callback에서 갱신
        self.local_position_received = False

        # yaw 추격(pursuit) 제어 상태
        self.yaw_cmd = 0.0  # 지금 명령 중인 목표 yaw (ramp로 서서히 수렴시킴)
        self._heading_warned = False  # heading 필드 없을 때 경고 로그 1회만 찍기 위한 플래그

        # OFFBOARD/ARM 진입 재시도 카운터
        self._mode_retry_tick = 0
        self.offboard_setpoint_counter = 0

        self.land_cmd_sent = False
        self.disarm_sent = False  # DISARM_ALTITUDE 이하로 내려왔을 때 강제 disarm 보냈는지

        # 고도 조건은 만족했는데 정렬이 안 맞아 LAND 진입을 못 하고 있는 시간 추적용.
        # 상태(APPROACH/DESCEND) 전환과는 완전히 무관하게, "현재 고도 < LAND_ALTITUDE"
        # 여부만으로 update_state_machine()에서 관리한다.
        self._land_stuck_since = None

        # 상태줄 갱신 스로틀용 카운터
        self._status_tick = 0

        self.state = STATE_SEARCH

        # ---- Subscribers ----
        self.aruco_sub = self.create_subscription(
            ArucoDetection,
            "/aruco_detections",
            self.aruco_callback,
            10,  # 이건 일반 ROS2 토픽이라 기본 QoS로 충분
        )

        self.status_sub = self.create_subscription(
            VehicleStatus,
            "/fmu/out/vehicle_status_v1",
            self.status_callback,
            PX4_QOS,
        )

        self.local_pos_sub = self.create_subscription(
            VehicleLocalPosition,
            "/fmu/out/vehicle_local_position_v1",
            self.local_position_callback,
            PX4_QOS,
        )

        # ---- Publishers ----
        self.offboard_pub = self.create_publisher(
            OffboardControlMode,
            "/fmu/in/offboard_control_mode",
            PX4_QOS,
        )

        self.traj_pub = self.create_publisher(
            TrajectorySetpoint,
            "/fmu/in/trajectory_setpoint",
            PX4_QOS,
        )

        self.cmd_pub = self.create_publisher(
            VehicleCommand,
            "/fmu/in/vehicle_command",
            PX4_QOS,
        )

        self.timer = self.create_timer(TIMER_PERIOD, self.timer_callback)  # 50Hz

        self.get_logger().info("Precision Landing Node 시작 (uXRCE-DDS)")

    # ------------------------------------------------------------------
    # Callbacks
    # ------------------------------------------------------------------

    def aruco_callback(self, msg):
        if len(msg.markers) == 0:
            self.marker_visible = False
            return

        marker = msg.markers[0]
        raw_x = marker.pose.position.x
        raw_y = marker.pose.position.y
        raw_z = marker.pose.position.z

        if self.marker_ever_seen:
            # 저역통과 필터(EMA): ArUco 포즈 추정의 프레임 단위 떨림을 완화.
            a = MARKER_FILTER_ALPHA
            self.marker_x = a * raw_x + (1 - a) * self.marker_x
            self.marker_y = a * raw_y + (1 - a) * self.marker_y
            self.marker_z = a * raw_z + (1 - a) * self.marker_z
        else:
            # 첫 감지는 필터링 없이 그대로 반영
            self.marker_x = raw_x
            self.marker_y = raw_y
            self.marker_z = raw_z

        self.marker_visible = True
        self.marker_ever_seen = True
        self.last_marker_time = self.get_clock().now()

    def status_callback(self, msg):
        self.arming_state = msg.arming_state
        self.nav_state = msg.nav_state

    def local_position_callback(self, msg):
        # PX4 local position은 NED: z가 아래로 양수. 고도는 -z.
        self.current_altitude = -msg.z

        heading = getattr(msg, "heading", None)
        if heading is None:
            if not self._heading_warned:
                self.get_logger().error(
                    "VehicleLocalPosition에 'heading' 필드가 없음 -> "
                    "yaw 추격이 heading=0.0으로 고정됨. px4_msgs 버전 확인 필요 "
                    "(예: `ros2 interface show px4_msgs/msg/VehicleLocalPosition`)"
                )
                self._heading_warned = True
            heading = 0.0
        self.current_heading = heading

        if not self.local_position_received:
            self.yaw_cmd = heading
        self.local_position_received = True

    def log_event(self, text, color=COLOR_RESET):
        sys.stdout.write("\n" + color + text + COLOR_RESET + "\n")
        sys.stdout.flush()

    def status_bar(self):
        err = self.horizontal_error() if self.marker_visible else 0.0
        lost = self.marker_is_stale()
        marker_color = COLOR_RED if lost else COLOR_GREEN
        marker_text = "LOST" if lost else "FOUND"

        offboard = (self.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD)
        armed = (self.arming_state == VehicleStatus.ARMING_STATE_ARMED)
        px4_color = COLOR_GREEN if (offboard and armed) else COLOR_RED
        px4_text = ("O" if offboard else "-") + ("A" if armed else "-")

        core = (
            f"[{self.state:^8}] "
            f"[{marker_color}{marker_text:^5}{COLOR_RESET}] "
            f"[{px4_color}{px4_text:^2}{COLOR_RESET}] "
            f"ALT:{self.current_altitude:5.2f}m "
            f"ERR:{err:4.2f}m"
        )
        v_part = f" V:({self.vx:+4.2f},{self.vy:+4.2f},{self.vz:+4.2f})"
        yaw_part = f" YAW:{math.degrees(self.yaw_cmd):6.1f}°"

        columns = shutil.get_terminal_size(fallback=(100, 24)).columns
        visible_len = lambda s: len(_ANSI_RE.sub("", s))

        parts = [core, v_part, yaw_part]
        while len(parts) > 1 and visible_len("".join(parts)) > columns - 1:
            parts.pop()

        line = "\r" + "".join(parts)

        sys.stdout.write("\033[2K\r" + line)
        sys.stdout.flush()

    # ------------------------------------------------------------------
    # Command helpers
    # ------------------------------------------------------------------

    def publish_vehicle_command(self, command, param1=0.0, param2=0.0, param3=0.0):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.param1 = float(param1)
        msg.param2 = float(param2)
        msg.param3 = float(param3)
        msg.command = command
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def publish_offboard_control_mode(self):
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = True
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        self.offboard_pub.publish(msg)

    def publish_trajectory_setpoint(self, vx, vy, vz, yaw=float("nan")):
        msg = TrajectorySetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = [float("nan"), float("nan"), float("nan")]
        msg.velocity = [vx, vy, vz]
        msg.yaw = yaw
        self.traj_pub.publish(msg)

    # ------------------------------------------------------------------
    # 좌표 변환: ArUco 카메라 좌표 -> NED 속도 명령
    # ------------------------------------------------------------------

    def compute_position_command(self):
        """마커까지의 body frame 오프셋을 현재 heading으로 회전시켜 world(NED) vx/vy로.

        yaw_cmd(명령 yaw)가 아니라 실제 current_heading으로 회전시킨다 — 기체가
        지금 실제로 향하고 있는 방향 기준으로 옆이동을 포함한 최단 경로를 낸다.

        [2026-09-08 수정] body_y는 원래 +self.marker_x였음 - 이 로직을 이식한
        search_phase_node.py에서 실비행 2회(서로 다른 yaw ~0°/~180°)로 테스트한
        결과, yaw가 달라도 매번 World_Y축(=body_y가 담당하는 축)만 계속 5m대
        진폭으로 진동하는 게 재현됨. 실패하는 축이 yaw와 무관하게 항상 body_y
        쪽이라는 건 회전 공식이 아니라 이 부호 자체가 원인이라는 뜻이라, 뒤집어서
        수정. 이 원본 파일은 그동안 이 버그를 안고 있었을 가능성이 있음 - 부호
        수정 전 상태로 실비행 검증된 적이 없어서 실제 영향은 미확인.
        """
        body_x = -self.marker_y
        body_y = -self.marker_x
        err = self.horizontal_error()
        if err < 1e-6:
            return 0.0, 0.0

        speed = min(MAX_XY_SPEED, KP_XY * err)
        ux, uy = body_x / err, body_y / err
        vx_body, vy_body = ux * speed, uy * speed

        ch, sh = math.cos(self.current_heading), math.sin(self.current_heading)
        vx_world = vx_body * ch - vy_body * sh
        vy_world = vx_body * sh + vy_body * ch
        return vx_world, vy_world

    def horizontal_error(self):
        return (self.marker_x ** 2 + self.marker_y ** 2) ** 0.5

    @staticmethod
    def ramp_toward(current, target, dt, max_accel=MAX_XY_ACCEL):
        max_delta = max_accel * dt
        delta = target - current
        if delta > max_delta:
            delta = max_delta
        elif delta < -max_delta:
            delta = -max_delta
        return current + delta

    # ------------------------------------------------------------------
    # 상태 머신
    # ------------------------------------------------------------------

    def marker_is_stale(self):
        if not self.marker_ever_seen:
            return True
        elapsed = (self.get_clock().now() - self.last_marker_time).nanoseconds / 1e9
        return elapsed > MARKER_LOST_TIMEOUT

    def log_marker_presence(self):
        stale = self.marker_is_stale()

        if not stale and not self.marker_confirmed:
            self.log_event("마커 확인", COLOR_GREEN)
            self.marker_confirmed = True

        elif stale and self.marker_confirmed:
            if self.state != STATE_LAND:
                self.log_event("마커 유실", COLOR_RED)
            self.marker_confirmed = False

    def update_state_machine(self):
        self.log_marker_presence()

        if self.marker_is_stale():
            if self.state == STATE_LAND:
                return

            if self.local_position_received and self.current_altitude < LAND_ALTITUDE:
                # 이미 착륙 고도 밑으로 내려온 상태에서 마커까지 놓치면, 다시 잡힐
                # 때까지 무한정 호버링하며 기다리는 게 더 위험/비효율적이다.
                # 저고도에서는 마커 없이도 그냥 수직으로 착륙을 강행한다.
                self.log_event(
                    f"저고도({LAND_ALTITUDE:.2f}m 미만)에서 마커 유실 -> 착륙 강행",
                    COLOR_RED,
                )
                self.state = STATE_LAND
                self._land_stuck_since = None
                return

            self.vx = self.ramp_toward(self.vx, 0.0, TIMER_PERIOD)
            self.vy = self.ramp_toward(self.vy, 0.0, TIMER_PERIOD)
            self.vz = self.ramp_toward(self.vz, 0.0, TIMER_PERIOD)
            self.state = STATE_SEARCH
            self._land_stuck_since = None
            return

        err = self.horizontal_error()

        # --- 고도 기반 stuck 타이머: APPROACH<->DESCEND 진동과 무관하게 독립 추적 ---
        # "현재 고도가 이미 LAND_ALTITUDE 아래인가"만 기준으로 삼는다. 예전 버전은
        # 이 타이머를 APPROACH 진입 때마다 None으로 리셋해버려서, err가 흔들려
        # APPROACH/DESCEND를 왔다갔다 하면 절대 누적되지 않는 죽은 코드였다.
        altitude_ok = (
            self.local_position_received and self.current_altitude < LAND_ALTITUDE
        )
        if altitude_ok:
            if self._land_stuck_since is None:
                self._land_stuck_since = self.get_clock().now()
            stuck_elapsed = (
                self.get_clock().now() - self._land_stuck_since
            ).nanoseconds / 1e9
        else:
            self._land_stuck_since = None
            stuck_elapsed = 0.0

        if self.state in (STATE_APPROACH, STATE_DESCEND) and altitude_ok:
            aligned = err < CENTER_TOLERANCE
            timed_out = stuck_elapsed > LAND_STUCK_TIMEOUT_SEC
            if aligned or timed_out:
                reason = "정렬됨" if aligned else f"{stuck_elapsed:.1f}s 정체 -> 강행"
                self.log_event(
                    f"고도 {LAND_ALTITUDE:.1f}m 미만 ({reason}) -> 착륙", COLOR_GREEN
                )
                self.state = STATE_LAND

        if self.state == STATE_SEARCH:
            self.state = STATE_APPROACH

        if self.state == STATE_APPROACH:
            target_vx, target_vy = self.compute_position_command()
            self.vx = self.ramp_toward(self.vx, target_vx, TIMER_PERIOD)
            self.vy = self.ramp_toward(self.vy, target_vy, TIMER_PERIOD)
            # 고도가 이미 낮은 상태에서 APPROACH로 다시 튕겨도 vz를 0으로 뚝 끊지 않고
            # 서서히 감쇠시켜 하강-정지 반복(채터링)을 줄인다.
            self.vz = self.ramp_toward(self.vz, 0.0, TIMER_PERIOD)
            if err < CENTER_TOLERANCE:
                self.state = STATE_DESCEND

        elif self.state == STATE_DESCEND:
            target_vx, target_vy = self.compute_position_command()
            self.vx = self.ramp_toward(self.vx, target_vx, TIMER_PERIOD)
            self.vy = self.ramp_toward(self.vy, target_vy, TIMER_PERIOD)
            self.vz = DESCEND_SPEED  # NED라 아래로 이동 = 양수

            if err > CENTER_TOLERANCE * 2:
                self.state = STATE_APPROACH

            # (참고) 위쪽 공용 stuck-timer 블록에서 이미 altitude_ok+aligned/timed_out
            # 조건으로 STATE_LAND 전환을 처리하므로 여기서 별도 분기는 필요 없다.

        elif self.state == STATE_LAND:
            # NAV_LAND로 PX4 자체 AUTO.LAND에 넘기지 않는다. 넘기는 순간 PX4의
            # 착륙 속도 파라미터(MPC_LAND_SPEED, 보통 DESCEND_SPEED보다 훨씬 빠름)로
            # 바뀌면서 "갑자기 너무 빨리 착륙"하는 원인이 됐다. 대신 끝까지 우리가
            # 직접 저속으로 하강시켜서 DISARM_ALTITUDE에서 disarm한다.
            self.vx = self.ramp_toward(self.vx, 0.0, TIMER_PERIOD)
            self.vy = self.ramp_toward(self.vy, 0.0, TIMER_PERIOD)
            self.vz = self.ramp_toward(self.vz, LAND_FINAL_SPEED, TIMER_PERIOD)

            if not self.land_cmd_sent:
                self.log_event("최종 착륙 하강 시작", COLOR_GREEN)
                self.land_cmd_sent = True

            if (
                not self.disarm_sent
                and self.local_position_received
                and self.current_altitude <= DISARM_ALTITUDE
            ):
                self.vx = 0.0
                self.vy = 0.0
                self.vz = 0.0
                self.publish_vehicle_command(
                    VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                    param1=0.0,  # 0.0 = disarm
                )
                self.log_event("고도 0.05m 이하 -> 강제 disarm", COLOR_GREEN)
                self.disarm_sent = True

    # ------------------------------------------------------------------
    # OFFBOARD/ARM 진입 (확인될 때까지 재시도)
    # ------------------------------------------------------------------

    def ensure_offboard_and_armed(self):
        # NAV_LAND를 더 이상 쓰지 않으므로(자체 저속 하강으로 대체) PX4가 스스로
        # 모드를 바꿀 일이 없다 -> disarm 전까지는 계속 OFFBOARD+ARMED를 유지해야 하고,
        # 여기서 특별히 재시도를 멈출 이유가 없다.
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
                VehicleCommand.VEHICLE_CMD_DO_SET_MODE,
                param1=1.0,
                param2=6.0,
            )
        if not armed:
            self.publish_vehicle_command(
                VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM,
                param1=1.0,
            )

    # ------------------------------------------------------------------
    # Main timer loop
    # ------------------------------------------------------------------

    def timer_callback(self):
        try:
            self.publish_offboard_control_mode()

            self.update_state_machine()
            self.publish_trajectory_setpoint(self.vx, self.vy, self.vz, yaw=self.yaw_cmd)

            self._status_tick += 1
            if self._status_tick >= STATUS_REFRESH_EVERY_N_TICKS:
                self._status_tick = 0
                self.status_bar()

            if self.offboard_setpoint_counter < 10:
                self.offboard_setpoint_counter += 1
                return

            self.ensure_offboard_and_armed()
        except Exception:
            sys.stdout.write("\n")
            sys.stdout.flush()
            self.get_logger().error(
                "timer_callback에서 예외 발생 (노드는 계속 실행됨):\n"
                + traceback.format_exc()
            )


def main(args=None):
    rclpy.init(args=args)
    node = PrecisionLanding()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        sys.stdout.write("\n")
        sys.stdout.flush()
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()


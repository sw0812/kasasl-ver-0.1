#!/usr/bin/env python3
"""ArUco 마커(고정 기준점) -> PX4 vehicle_visual_odometry 브릿지.

목적: GPS/광류가 전혀 없는 실내에서, 위치가 미리 알려진 고정 기준
마커(기본: search_phase_node.py의 VERTIPORT_MARKER_ID=0, world (0,0,0))를
카메라로 보고 그 상대위치를 뒤집어 "드론이 world 좌표계에서 어디 있는지"를
계산해 /fmu/in/vehicle_visual_odometry로 PX4 EKF에 공급한다.

주의 (반드시 실제 환경에서 검증할 것 — landing.py/pitchstoplanding.py와
동일한 원칙):
  - 카메라 완전 하향(pitch 90도) 고정 장착 가정, 짐벌 없음.
  - 카메라->body 변환은 landing.py에서 실비행으로 검증된 것과 동일한 매핑
    (body_x=-cam_y, body_y=cam_x, body_z=cam_z)을 그대로 재사용.
  - 카메라가 body 원점에서 떨어진 오프셋(cam_offset_*)은 기본 0으로 두되,
    실측 필요하면 파라미터로 보정할 것.
  - 기준 마커의 world yaw(ref_marker_yaw_deg)는 기본 0으로 두었으나 실제
    마커가 인쇄된 방향과 world 좌표축이 어떻게 정렬되는지 반드시 확인 후
    맞춰야 한다 — 안 맞으면 위치는 맞아도 heading이 틀어져서 EKF가 이상하게
    수렴할 수 있음.
  - 이 노드는 position(+orientation)만 채워서 보낸다(velocity는 NaN =
    미공급). PX4 EKF2_EV_CTRL에서 "velocity" 비트는 끄고 position/yaw만
    켜는 걸 권장.
  - PX4 쪽 EKF2_EV_CTRL 등 실제 융합 파라미터는 이 스크립트가 건드리지
    않는다 — 안전상 반드시 벤치에서 값이 그럴듯하게 나오는지 먼저 확인한
    뒤에 사람이 직접 켤 것.

실행 예:
  python3 aruco_visual_odom.py \\
      --ros-args -p ref_marker_id:=0 -p ref_marker_x:=0.0 -p ref_marker_y:=0.0 \\
      -p ref_marker_z:=0.0 -p ref_marker_yaw_deg:=0.0
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy

from aruco_opencv_msgs.msg import ArucoDetection
from px4_msgs.msg import VehicleOdometry

# landing.py/pitchstoplanding.py에서 실비행으로 검증된 카메라->body 매핑과
# 동일: body_x=-cam_y, body_y=cam_x, body_z=cam_z (카메라 광축=body 아래축).
R_BODY_FROM_CAM = np.array([
    [0.0, -1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 0.0, 1.0],
])


def quat_to_rotmat(w, x, y, z):
    n = (w * w + x * x + y * y + z * z) ** 0.5
    if n < 1e-9:
        return np.eye(3)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def rotmat_to_quat(rm):
    tr = np.trace(rm)
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (rm[2, 1] - rm[1, 2]) * s
        y = (rm[0, 2] - rm[2, 0]) * s
        z = (rm[1, 0] - rm[0, 1]) * s
    else:
        i = np.argmax([rm[0, 0], rm[1, 1], rm[2, 2]])
        if i == 0:
            s = 2.0 * np.sqrt(1.0 + rm[0, 0] - rm[1, 1] - rm[2, 2])
            w = (rm[2, 1] - rm[1, 2]) / s
            x = 0.25 * s
            y = (rm[0, 1] + rm[1, 0]) / s
            z = (rm[0, 2] + rm[2, 0]) / s
        elif i == 1:
            s = 2.0 * np.sqrt(1.0 + rm[1, 1] - rm[0, 0] - rm[2, 2])
            w = (rm[0, 2] - rm[2, 0]) / s
            x = (rm[0, 1] + rm[1, 0]) / s
            y = 0.25 * s
            z = (rm[1, 2] + rm[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + rm[2, 2] - rm[0, 0] - rm[1, 1])
            w = (rm[1, 0] - rm[0, 1]) / s
            x = (rm[0, 2] + rm[2, 0]) / s
            y = (rm[1, 2] + rm[2, 1]) / s
            z = 0.25 * s
    return np.array([w, x, y, z])


class ArucoVisualOdom(Node):
    def __init__(self):
        super().__init__("aruco_visual_odom")

        self.declare_parameter("ref_marker_id", 0)
        self.declare_parameter("ref_marker_x", 0.0)
        self.declare_parameter("ref_marker_y", 0.0)
        self.declare_parameter("ref_marker_z", 0.0)
        self.declare_parameter("ref_marker_yaw_deg", 0.0)
        self.declare_parameter("cam_offset_x", 0.0)
        self.declare_parameter("cam_offset_y", 0.0)
        self.declare_parameter("cam_offset_z", 0.0)
        self.declare_parameter("position_variance", 0.02)
        self.declare_parameter("orientation_variance", 0.05)

        self.ref_id = self.get_parameter("ref_marker_id").value
        self.ref_pos = np.array([
            self.get_parameter("ref_marker_x").value,
            self.get_parameter("ref_marker_y").value,
            self.get_parameter("ref_marker_z").value,
        ])
        yaw = math.radians(self.get_parameter("ref_marker_yaw_deg").value)
        Rz = np.array([
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ])
        # 마커가 바닥에 인쇄면이 위(하늘)를 보게 놓여있다고 가정 -> 마커
        # 자신의 +Z축(인쇄면에서 튀어나오는 방향, ArUco 표준 정의)은 world
        # 기준 "위쪽"을 향함. NED는 Z가 아래(+)라서 마커 Z축은 world -Z
        # 방향 -> 그대로 Rz만 쓰면 카메라 위치가 마커 "아래"로 계산되는
        # 부호 오류가 생김(2026-09-21 실측으로 발견). X는 유지, Y/Z를
        # 뒤집어 마커 Z축이 world 아래(NED +Z)가 아니라 위를 향하게 맞춤.
        flip = np.diag([1.0, -1.0, -1.0])
        self.R_world_marker = Rz @ flip
        self.cam_offset_body = np.array([
            self.get_parameter("cam_offset_x").value,
            self.get_parameter("cam_offset_y").value,
            self.get_parameter("cam_offset_z").value,
        ])
        self.pos_var = float(self.get_parameter("position_variance").value)
        self.ori_var = float(self.get_parameter("orientation_variance").value)

        # search_phase_node.py/landing.py의 PX4_QOS와 동일하게 맞춤 - durability를
        # TRANSIENT_LOCAL로 뒀더니 Micro XRCE-DDS가 이 퍼블리셔를 uxrce_dds_client
        # 쪽에 아예 안 이어줘서(2026-09-21 실측: "listener vehicle_visual_odometry"
        # -> never published) VOLATILE로 바꿈.
        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.create_subscription(ArucoDetection, "/aruco_detections", self._on_detection, 10)
        self.odom_pub = self.create_publisher(VehicleOdometry, "/fmu/in/vehicle_visual_odometry", qos)

        self.get_logger().info(
            f"[aruco_visual_odom] 기준 마커 id={self.ref_id}, world pos={self.ref_pos.tolist()}, "
            f"yaw={self.get_parameter('ref_marker_yaw_deg').value}deg — 이 값들이 실제와 "
            "안 맞으면 EKF에 잘못된 위치를 먹이게 되니 반드시 검증할 것."
        )

    def _on_detection(self, msg: ArucoDetection):
        target = None
        for m in msg.markers:
            if m.marker_id == self.ref_id:
                target = m
                break
        if target is None:
            return

        p = target.pose.position
        o = target.pose.orientation
        t_cam_marker = np.array([p.x, p.y, p.z])
        R_cam_marker = quat_to_rotmat(o.w, o.x, o.y, o.z)

        # T_cam_marker의 역변환 -> 카메라가 마커 기준으로 어디/어떤 자세인지
        R_marker_cam = R_cam_marker.T
        t_marker_cam = -R_marker_cam @ t_cam_marker

        # 마커의 world pose와 합성 -> 카메라의 world pose
        R_world_cam = self.R_world_marker @ R_marker_cam
        t_world_cam = self.R_world_marker @ t_marker_cam + self.ref_pos

        # 카메라->body 고정 외부파라미터 합성 -> body(드론)의 world pose
        # (world<-body = world<-cam * cam<-body, cam<-body = (body<-cam)^-1)
        R_cam_body = R_BODY_FROM_CAM.T
        t_cam_body = -R_cam_body @ self.cam_offset_body
        R_world_body = R_world_cam @ R_cam_body
        t_world_body = R_world_cam @ t_cam_body + t_world_cam

        q = rotmat_to_quat(R_world_body)

        odom = VehicleOdometry()
        odom.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        odom.timestamp_sample = odom.timestamp
        odom.pose_frame = VehicleOdometry.POSE_FRAME_NED
        odom.position = [float(t_world_body[0]), float(t_world_body[1]), float(t_world_body[2])]
        odom.q = [float(q[0]), float(q[1]), float(q[2]), float(q[3])]
        odom.velocity_frame = VehicleOdometry.VELOCITY_FRAME_NED
        odom.velocity = [float("nan")] * 3       # 속도는 안 채움 - EKF2_EV_CTRL에서 velocity 비트는 끌 것
        odom.angular_velocity = [float("nan")] * 3
        odom.position_variance = [self.pos_var] * 3
        odom.orientation_variance = [self.ori_var] * 3
        # velocity_variance는 (position/velocity/q와 달리) msg 정의에 "NaN=무효"
        # 규정이 없음 - NaN을 넣으면 EKF2 수신측이 샘플 전체를 이상하게
        # 처리할 수 있어(2026-09-21 실측: NaN으로 보낸 뒤 cs_ev_pos가 계속
        # False로 안 켜짐) 그냥 큰 유한값(사실상 "안 믿음")으로 채움.
        odom.velocity_variance = [1e6] * 3
        odom.quality = 100
        self.odom_pub.publish(odom)


def main():
    rclpy.init()
    node = ArucoVisualOdom()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

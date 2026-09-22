#!/usr/bin/env python3
"""
마커가 카메라에 "보이는 동안 계속" 모터를 돌리고, 안 보이면 멈추는 스크립트.

MAV_CMD_ACTUATOR_TEST(310)를 씀 - disarm 상태에서만 동작하고 위치추정이
전혀 필요 없음. param2(timeout)를 짧게(예: 0.4초) 잡아서 마커가 보이는 동안
매 프레임(detection 콜백마다) 명령을 계속 갱신 발사한다 - PX4가 "이 시간
안에 새 명령이 안 오면 자동으로 정지"하도록 되어 있어서, 마커가 화면에서
사라지면 그 시점의 마지막 timeout이 만료되며 자동으로 멈춘다(별도 정지
명령 없이도 최대 timeout 시간 안에 정지).

사용법 (프로펠러 반드시 제거 후):
  python3 marker_motor_trigger.py --motor 1 --value 0.15
"""
import argparse

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from aruco_opencv_msgs.msg import ArucoDetection
from px4_msgs.msg import VehicleCommand

MAV_CMD_ACTUATOR_TEST = 310
REFRESH_TIMEOUT_SEC = 0.4  # 이 시간 안에 새 명령이 없으면 PX4가 자동 정지


class MarkerMotorTrigger(Node):
    def __init__(self, args):
        super().__init__("marker_motor_trigger")
        self.args = args
        self.was_visible = False

        qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.VOLATILE,
            history=HistoryPolicy.KEEP_LAST,
            depth=5,
        )
        self.cmd_pub = self.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)
        self.currently_visible = False
        self.create_subscription(ArucoDetection, "/aruco_detections", self._on_detection, 10)
        # detection 콜백(카메라 프레임레이트, 20~30Hz)에 맞춰 매번 발행하면
        # uXRCE-DDS 시리얼 채널이 밀려서(921600 baud) 명령이 아예 안 가는
        # 문제가 실측됨(2026-09-22) - 발행은 이 타이머(10Hz)에서만 하고,
        # 콜백은 "지금 보이는지" 상태만 갱신하도록 분리.
        self.create_timer(0.1, self._on_timer)
        self.get_logger().info(
            f"[marker_motor_trigger] marker_id={args.marker_id} 보이는 동안 계속 "
            f"모터 {args.motor}번을 value={args.value}로 회전, 안 보이면 "
            f"{REFRESH_TIMEOUT_SEC}초 이내 자동 정지. 프로펠러 반드시 제거된 상태에서만 사용할 것."
        )

    def _send_actuator_test(self, value, timeout):
        msg = VehicleCommand()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.command = MAV_CMD_ACTUATOR_TEST
        msg.param1 = float(value)
        msg.param2 = float(timeout)
        msg.param5 = float(self.args.motor)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        self.cmd_pub.publish(msg)

    def _on_detection(self, msg: ArucoDetection):
        self.currently_visible = any(
            (self.args.marker_id < 0 or m.marker_id == self.args.marker_id)
            for m in msg.markers
        )

    def _on_timer(self):
        visible = self.currently_visible
        if visible:
            # 10Hz로만 갱신 발사 - 마커가 계속 보이는 한 정지 타임아웃이 계속 미뤄짐
            self._send_actuator_test(self.args.value, REFRESH_TIMEOUT_SEC)
            if not self.was_visible:
                self.get_logger().info("마커 인식됨 -> 회전 시작")
        elif self.was_visible:
            # 즉시 정지 명령도 같이 보내서 최대 대기시간(0.4s) 없이 바로 멈추게 함
            self._send_actuator_test(float("nan"), 0.0)
            self.get_logger().info("마커 사라짐 -> 정지")
        self.was_visible = visible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--marker-id", type=int, default=-1, help="-1이면 아무 마커나 인식되면 회전")
    ap.add_argument("--motor", type=int, default=1)
    ap.add_argument("--value", type=float, default=0.15)
    args, _ = ap.parse_known_args()

    rclpy.init()
    node = MarkerMotorTrigger(args)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        # 종료 시 확실히 정지
        node._send_actuator_test(float("nan"), 0.0)
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

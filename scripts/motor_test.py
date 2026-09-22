#!/usr/bin/env python3
"""
MAV_CMD_ACTUATOR_TEST(310)로 개별 모터 하나를 짧게 돌려보는 벤치테스트 스크립트.
disarm 상태에서만 동작함(armed 중엔 PX4가 이 명령을 거부하도록 설계돼 있음 —
그래서 search_phase_node의 OFFBOARD arm 경로와 달리 위치추정이 전혀 필요 없음).

사용법 (프로펠러 반드시 제거 후):
  python3 motor_test.py --motor 1 --value 0.15 --duration 2.0
"""
import argparse
import time

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from px4_msgs.msg import VehicleCommand

MAV_CMD_ACTUATOR_TEST = 310


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--motor", type=int, default=1, help="모터 번호 1~4")
    ap.add_argument("--value", type=float, default=0.15,
                     help="출력값 0~1 (0=최소 회전, 1=최대). 처음엔 낮게 시작 권장")
    ap.add_argument("--duration", type=float, default=2.0, help="회전 지속 시간(초), 최대 3초")
    args = ap.parse_args()

    if not (1 <= args.motor <= 8):
        raise SystemExit("모터 번호는 1~8 사이여야 합니다")
    if not (0.0 <= args.value <= 1.0):
        raise SystemExit("value는 0~1 사이여야 합니다")
    duration = min(max(args.duration, 0.1), 3.0)

    rclpy.init()
    node = Node("motor_test")
    qos = QoSProfile(
        reliability=ReliabilityPolicy.BEST_EFFORT,
        durability=DurabilityPolicy.TRANSIENT_LOCAL,
        history=HistoryPolicy.KEEP_LAST,
        depth=1,
    )
    pub = node.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", qos)

    def send(value, timeout):
        msg = VehicleCommand()
        msg.timestamp = int(node.get_clock().now().nanoseconds / 1000)
        msg.command = MAV_CMD_ACTUATOR_TEST
        msg.param1 = float(value)          # NaN=정지, 0=최소 회전, 1=최대
        msg.param2 = float(timeout)        # 이 시간 뒤 자동으로 이전 값(정지)으로 복원
        msg.param5 = float(args.motor)     # ACTUATOR_OUTPUT_FUNCTION_MOTORn (n=모터번호)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        pub.publish(msg)

    print(f"모터 {args.motor}번 -> value={args.value}, {duration}초간 회전 시도")
    print("(PX4는 disarm 상태에서만 이 명령을 받아줍니다 — armed면 거부됨)")

    # 발행 즉시 사라지면 구독측(uxrce_dds_client)이 못 받을 수 있어 짧게 반복 전송
    t0 = time.time()
    while time.time() - t0 < 0.3:
        send(args.value, duration)
        time.sleep(0.05)

    time.sleep(duration)
    send(float("nan"), 0.0)  # 명시적으로 정지
    time.sleep(0.2)
    print("정지 명령 전송 완료")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

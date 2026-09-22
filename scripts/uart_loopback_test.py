#!/usr/bin/env python3
"""
UART 루프백/원인진단 테스트 (Jetson /dev/ttyTHS0, 8번=TX, 10번=RX)

사용법:
  python3 uart_loopback_test.py
  python3 uart_loopback_test.py --dev /dev/ttyTHS0 --baud 57600

테스트 방법:
  - 루프백 테스트: Pixhawk 떼거나 그대로 두고, Jetson 40핀 헤더 8번(TX)과
    10번(RX)을 점퍼선으로 직접 연결한 뒤 실행. 보낸 데이터가 그대로 돌아오면
    Jetson TX/RX 하드웨어 자체는 정상.
  - Pixhawk 연결 테스트: 점퍼 없이 Pixhawk에 연결된 상태로 실행하면, Pixhawk가
    보내는 데이터가 잡히는지(수신), 그리고 이쪽에서 보낸 게 Pixhawk에 영향을
    주는지 확인 가능 (단, Pixhawk 수신 여부는 uxrce_dds_client status 등으로
    별도 확인 필요).

결과와 log 파일(uart_loopback_test.log, 같은 폴더)에 원인 진단까지 자동으로 남습니다.
"""
import argparse
import datetime
import os
import struct
import sys
import termios
import time

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uart_loopback_test.log")


def log(msg, f):
    line = f"{msg}"
    print(line)
    f.write(line + "\n")


def open_raw(dev, baud):
    fd = os.open(dev, os.O_RDWR | os.O_NOCTTY | os.O_NONBLOCK)
    attrs = termios.tcgetattr(fd)
    baud_const = getattr(termios, f"B{baud}", None)
    if baud_const is None:
        raise ValueError(f"지원 안 하는 baud: {baud}")
    attrs[4] = baud_const  # ispeed
    attrs[5] = baud_const  # ospeed
    attrs[2] = termios.CS8 | termios.CLOCAL | termios.CREAD
    attrs[0] = 0
    attrs[1] = 0
    attrs[3] = 0
    attrs[6][termios.VMIN] = 0
    attrs[6][termios.VTIME] = 0
    termios.tcsetattr(fd, termios.TCSANOW, attrs)
    termios.tcflush(fd, termios.TCIOFLUSH)
    return fd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", default="/dev/ttyTHS0")
    ap.add_argument("--baud", type=int, default=57600)
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args()

    with open(LOG_PATH, "a") as f:
        log("", f)
        log("=" * 60, f)
        log(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] 테스트 시작", f)
        log(f"장치: {args.dev}  baud: {args.baud}", f)

        try:
            fd = open_raw(args.dev, args.baud)
        except Exception as e:
            log(f"[FAIL] 포트 열기/설정 실패: {e}", f)
            sys.exit(1)

        marker = b"PX4JETSON" + struct.pack("<I", int(time.time()) & 0xFFFFFFFF)
        total_sent = 0
        total_recv = b""

        for i in range(args.rounds):
            payload = marker + f"-{i:02d}".encode()
            os.write(fd, payload)
            total_sent += len(payload)
            time.sleep(0.3)
            try:
                chunk = os.read(fd, 4096)
            except BlockingIOError:
                chunk = b""
            total_recv += chunk
            log(f"  round {i}: 전송 {len(payload)}B, 누적 수신 {len(total_recv)}B", f)

        os.close(fd)

        log("-" * 60, f)
        log(f"총 전송: {total_sent} bytes", f)
        log(f"총 수신: {len(total_recv)} bytes", f)
        log("-" * 60, f)

        if len(total_recv) == 0:
            log("[진단] 수신 0바이트.", f)
            log("  -> 루프(TX->RX)가 열려있음(끊김). 원인 후보:", f)
            log("     - 8번(TX)과 10번(RX)이 실제로 점퍼로 연결 안 됨", f)
            log("     - Jetson TX(8번, UART1_TXD)가 실제로 신호를 안 내보냄", f)
            log("     - 포트/baud가 실제 배선과 안 맞음", f)
            log("  => Jetson TX(8번) 쪽 또는 점퍼선 자체에 결함 가능성 높음", f)
        elif marker[:10] in total_recv:
            log("[진단] 수신 데이터에 보낸 마커 패턴이 그대로 들어있음.", f)
            log("  -> Jetson TX(8번)이 실제로 라인을 구동하고 있고,", f)
            log("     Jetson RX(10번)도 정상 수신 중.", f)
            log("  -> Jetson 쪽 UART 하드웨어(양방향 모두)는 완전히 정상으로 확인됨.", f)
            log("  -> 이 상태에서 Pixhawk 연결 시에도 'disconnected'라면,", f)
            log("     문제는 Jetson이 아니라 Pixhawk까지 가는 TX 배선/커넥터,", f)
            log("     또는 Pixhawk RX 핀 쪽에 있음.", f)
        else:
            log("[진단] 일부 바이트는 수신됐지만 보낸 것과 그대로 일치하지 않음.", f)
            log("  -> baud 불일치, 노이즈, 또는 접촉 불량(간헐적 연결) 가능성.", f)
            log(f"  수신 앞부분(hex): {total_recv[:80].hex()}", f)

        log(f"[{datetime.datetime.now().isoformat(timespec='seconds')}] 테스트 종료, 로그 저장: {LOG_PATH}", f)


if __name__ == "__main__":
    main()

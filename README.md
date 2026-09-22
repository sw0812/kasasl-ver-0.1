# KASA SL — 정밀착륙 실기체 시스템 (ver 0.1)

Pixhawk 6X + Jetson(Orin/Xavier, ttyTHS0 UART) + USB 웹캠 기반, **GPS 없는 실내에서 ArUco 마커로 위치추정을 대체**해서 PX4 OFFBOARD 미션(탐색+정밀착륙)을 돌리는 실기체 제어 시스템.

2026-09-21~22 세션에서 처음으로 "미션 버튼 → ArUco 위치추정 → EKF 융합 → ARM → OFFBOARD → 이륙감지"까지 전 구간이 실증된 시점의 스냅샷입니다. 실비행(프로펠러 장착) 자체는 아직 검증 안 됐습니다 — 아래 "안전 경고" 꼭 읽어주세요.

## 하드웨어 구성
- **Pixhawk 6X** (또는 호환 FMU v6X) — Jetson과 40핀 헤더 UART(TX/RX 크로스)로 연결, USB는 진단/파라미터 설정용
- **Jetson** (Orin/Xavier 계열, L4T/Tegra) — UART가 `/dev/ttyTHS0`로 잡히는 보드
- USB 웹캠 (ArUco 마커 인식용, 아래를 보는 방향으로 장착 가정 — `body_x=-cam_y, body_y=cam_x` 컨벤션)
- ArUco `DICT_5X5_50`, ID 0 마커 (기준점, world 원점(0,0,0)에 고정) — 실제 인쇄 크기를 `control_panel/camera_calibration.yaml`/`aruco_tracker.yaml`의 `marker_size`와 일치시킬 것

## 소프트웨어 전제조건
- ROS2 Foxy
- `px4_msgs` — **이 리포의 PX4 펌웨어 버전과 정확히 맞는 브랜치로 빌드**되어 있어야 함(안 맞으면 토픽 파싱이 조용히 깨짐)
- `aruco_opencv` (`aruco_tracker_autostart`), `usb_cam`
- `MicroXRCEAgent` (uXRCE-DDS 에이전트, PATH에 있어야 함)
- `pymavlink`, `mavproxy` (`~/.local/bin/mavproxy.py`)
- Docker (QGroundControl을 컨테이너로 띄우는 경우만 — `control_panel/main.py`의 `qgc` 서브시스템 참고, 필수 아님)

## ⚠️ 실행 전 필수 PX4 파라미터 (QGC 또는 MAVLink로 설정)

이 파라미터들이 안 맞으면 겉보기엔 다 정상인데 위치추정/연결이 계속 안 됩니다 — 실측으로 확인된 것들이라 반드시 먼저 맞춰두세요.

| 파라미터 | 값 | 이유 |
|---|---|---|
| `UXRCE_DDS_CFG` | TELEM1 (보드마다 다름, 실측 필요) | Jetson UART가 물린 실제 TELEM 포트로 uXRCE-DDS 클라이언트를 켬 |
| `SER_TELx_BAUD` (TELEM1이면 `SER_TEL1_BAUD`) | `921600` | Jetson agent와 baud 일치 |
| `MAV_x_CONFIG` (같은 포트를 쓰던 MAVLink 인스턴스) | `Disabled` | 같은 UART에 MAVLink와 uXRCE-DDS를 동시에 못 물림 |
| `EKF2_HGT_REF` | `Vision`(3) | GPS 없으므로 고도 기준을 vision으로 — 기본값(GPS)로 두면 고도가 계속 발산함 |
| `EKF2_EV_CTRL` | `11` (수평위치+수직위치+yaw, 속도 비트는 끔) | ArUco 기반 위치를 EKF에 융합 |
| `BAT1_N_CELLS` | 실제 배터리 셀 수(예: 3S면 `3`) | **0으로 두면 정상 전압도 "Emergency battery"로 오판해서 ARM이 계속 거부됨** |

**파라미터를 스크립트/MAVLink로 직접 설정할 때 주의**: 정수형(INT32) 파라미터는 `float(정수)`로 그냥 형변환해서 보내면 안 됩니다 — MAVLink 스펙상 정수 비트패턴을 그대로 float32 슬롯에 재해석(`struct.pack('<i',v)` → `struct.unpack('<f',...)`)해서 보내야 합니다. 이걸 안 지키면 파라미터가 조용히 쓰레기값으로 저장되고, `param show`(PX4 NSH 셸)로 봐야만 진짜 값을 확인할 수 있습니다.

파라미터 변경은 대부분 `reboot_required` — 재부팅은 **USB(MAVLink)가 아니라 UART/uXRCE-DDS 경유로 보낼 것** (USB로 재부팅 명령을 보내면 정상 부팅이 아니라 부트로더로 빠지는 경향이 실측됨).

### USB가 부트로더(`PX4 BL FMU`)에 갇혔을 때
재플래시 없이 탈출 가능:
```python
import serial, time
s = serial.Serial('/dev/ttyACM0', 115200, timeout=2)
s.write(bytes([0x21, 0x20]))  # GET_SYNC + EOC
time.sleep(0.3); print(s.read(64).hex())  # 0x12 0x13... 응답 오면 부트로더 살아있음
s.write(bytes([0x30, 0x20]))  # PROTO_BOOT + EOC -> 앱으로 강제 점프
```

## 디렉터리 구조
```
control_panel/       KASA 정밀착륙 통합 GUI(main.py, tkinter) + 실행 스크립트
  main.py              xrce/camera/aruco/qgc/landing/mission 토글, 텔레메트리, ArUco 라이브뷰
  mavproxy_supervisor.sh   USB(/dev/pixhawk) <-> UDP 14550/14551, USB 재연결 자동복구
  run_main.sh          ROS2 환경 소싱 + main.py 실행 (+ X서버 크래시 자동재시작)
  99-pixhawk.rules     udev 규칙: Pixhawk 6X를 항상 /dev/pixhawk로 심볼릭 링크
  camera_calibration.yaml
ros2_nodes/           단독 실행 가능한 ROS2 노드 스크립트 (my_first_pkg에서 추출)
  search_phase_node.py   탐색(그리드 순회) + 착륙 통합 미션 노드, OFFBOARD
  aruco_visual_odom.py   ★ GPS-denied 핵심 — ArUco 마커 역산 -> vehicle_visual_odometry
  marker_motor_trigger.py  마커 보이는 동안 모터 회전(disarm 상태, 벤치테스트용)
  line_tracker.py        비전 라인트레이싱 유틸(search_phase_node 의존성)
precision_landing/
  landing.py            착륙 단독 실행용(search_phase_node의 착륙 로직과 동일 계열)
scripts/
  uart_loopback_test.py   Jetson UART TX/RX 배선 자체 점검(점퍼 루프백)
  motor_test.py           단발성 모터 회전 테스트(disarm, MAV_CMD_ACTUATOR_TEST)
desktop_launchers/    Ubuntu 바탕화면 아이콘(.desktop) — 절대경로가 이 리포 원본 사용자 기준이라 옮긴 뒤 Exec= 경로 수정 필요
```

## 실행 순서 (다른 기기에서 처음 켤 때)

1. **udev 규칙 설치**: `sudo cp control_panel/99-pixhawk.rules /etc/udev/rules.d/ && sudo udevadm control --reload-rules`
2. **PX4 파라미터**: 위 표대로 QGC에서 맞추고 재부팅(UART 경유 권장)
3. **ROS2 환경 소싱**: `source /opt/ros/foxy/setup.bash && source ~/ros2_ws/install/setup.bash && export ROS_DOMAIN_ID=0`
4. **uXRCE-DDS 에이전트**: `MicroXRCEAgent serial --dev /dev/ttyTHS0 -b 921600` (자동시작이 안 붙으면 PX4 셸에서 `uxrce_dds_client stop` 후 `uxrce_dds_client start -t serial -d /dev/ttyS6 -b 921600`로 강제 재시작 — 포트 번호는 보드마다 다를 수 있음, `ttyS0`~`ttyS7`을 돌며 MAVLink 테스트로 실측 필요)
5. **카메라 + ArUco**:
   ```
   ros2 run usb_cam usb_cam_node_exe --ros-args -p video_device:=/dev/video0 -r image_raw:=/camera/image_raw -r camera_info:=/camera/camera_info
   ros2 run aruco_opencv aruco_tracker_autostart --ros-args --params-file <aruco_tracker.yaml>
   ```
6. **위치추정 브릿지** (마커가 계속 카메라에 보여야 함): `python3 ros2_nodes/aruco_visual_odom.py --ros-args -p ref_marker_id:=0`
7. **제어판**: `control_panel/run_main.sh` (또는 `python3 control_panel/main.py`) → GUI에서 xrce/camera/aruco 토글 켜고 마커 인식 확인 → "미션(탐색+착륙)" 버튼

각 재부팅/재연결 뒤에는 4~6번(에이전트, 브릿지)을 새로 띄워야 새 PX4 세션에 확실히 붙습니다(오래된 프로세스는 살아있어도 새 세션에 자동으로 안 갈아탈 수 있음 — `listener vehicle_visual_odometry`(PX4 NSH 셸)로 "최근 몇 초 이내"인지 확인 권장).

## 안전 경고
- **프로펠러를 뗀 상태에서만 ARM/모터 테스트를 하세요.** 이 스냅샷은 벤치(고정)에서만 검증됐고 실비행 중 마커 유실 시 동작(자동 호버/착륙 등)은 검증 안 됐습니다.
- `search_phase_node.py`/`landing.py`의 `mission`/`landing`은 실제로 ARM+OFFBOARD 제어 명령을 내보냅니다.
- 벤치 모터 테스트는 `MAV_CMD_ACTUATOR_TEST`(disarm 전용, 위치추정 불필요) 방식(`scripts/motor_test.py`, `ros2_nodes/marker_motor_trigger.py`)을 쓰세요 — 굳이 ARM까지 갈 필요 없습니다.

## 알려진 이슈 / 미해결
- `uxrce_dds_client` 부팅 자동시작이 가끔 안 붙어서 수동 재시작이 필요할 수 있음(포트별 baud 파라미터 오염 이력 있음 — `param show SER_TELx_BAUD`로 진짜 값 확인 권장)
- USB가 왜 애초에 부트로더로 빠지는지 근본원인 미해결(위 탈출법은 대증요법)
- `aruco_visual_odom.py`의 카메라 오프셋(`cam_offset_*`)·마커 yaw(`ref_marker_yaw_deg`)는 기본값(0) — 실제 장착 환경에 맞게 실측 보정 필요
- GUI(`main.py`)에 `aruco_visual_odom.py`가 서브시스템으로 통합돼 있지 않음(수동 실행)

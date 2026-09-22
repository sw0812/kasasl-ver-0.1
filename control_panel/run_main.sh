#!/bin/bash
source /opt/ros/foxy/setup.bash
source /home/sasllab/ros2_ws/install/setup.bash
export ROS_DOMAIN_ID=0
export DISPLAY="${DISPLAY:-:1}"
cd "$(dirname "$0")"

# 이 Jetson의 X 서버가 드물게 RenderAddGlyphs에서 BadLength로 죽는 버그가
# 있어서(완전히 못 없앰, 근본 원인은 X 서버/드라이버 쪽) 죽으면 바로
# 자동으로 재시작해서 사용자가 다시 켤 필요 없게 함.
while true; do
    python3 main.py
    code=$?
    if [ $code -eq 0 ]; then
        break  # 창을 정상적으로 닫은 경우(X 종료 X 아님) 재시작 안 함
    fi
    echo "[run_main.sh] main.py 비정상 종료(exit $code) - 서브프로세스 정리 후 2초 뒤 재시작"
    # main.py가 죽으면 띄워뒀던 서브프로세스들이 고아로 남아 다음 인스턴스와
    # 충돌(포트/장치 중복 점유)할 수 있어 재시작 전에 정리한다.
    pkill -f "MicroXRCEAgent" 2>/dev/null
    pkill -f "usb_cam_node_exe" 2>/dev/null
    pkill -f "aruco_tracker_autostart" 2>/dev/null
    sleep 2
done

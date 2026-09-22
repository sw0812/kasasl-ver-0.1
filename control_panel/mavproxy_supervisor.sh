#!/bin/bash
# Pixhawk USB 재연결/전원 재인가 때마다 ttyACM 번호가 바뀌면서 mavproxy가
# 죽은 포트를 계속 붙잡는 문제 때문에 만든 감시/자동재시작 스크립트.
#
# 전제: /etc/udev/rules.d/99-pixhawk.rules 로 /dev/pixhawk 심볼릭 링크가
# 설정되어 있어야 함 (udev가 ttyACM 번호와 무관하게 항상 같은 이름을 붙여줌).
#
# 동작:
#   1. /dev/pixhawk 가 나타날 때까지 대기
#   2. mavproxy를 /dev/pixhawk로 실행
#   3. 백그라운드에서 2초마다 /dev/pixhawk가 실제로 가리키는 장치가
#      바뀌었는지 감시(USB 재연결로 ttyACM 번호가 바뀌면 링크 타겟도
#      바뀜) -> 바뀌면 mavproxy를 죽여서 재시작 유도
#   4. mavproxy가 (감시에 의해서든, 자체적으로든) 죽으면 1번으로 복귀

DEV=/dev/pixhawk
BAUD=57600
OUT1=udp:127.0.0.1:14550
OUT2=udp:127.0.0.1:14551
MAVPROXY=/home/sasllab/.local/bin/mavproxy.py

log() { echo "[$(date '+%H:%M:%S')] $*"; }

while true; do
    while [ ! -e "$DEV" ]; do
        log "대기 중: $DEV 없음 (Pixhawk USB 연결 확인 필요)"
        sleep 2
    done

    target=$(readlink -f "$DEV")
    log "연결 시도: $DEV -> $target"

    python3 "$MAVPROXY" --master="$DEV" --baudrate="$BAUD" \
        --out="$OUT1" --out="$OUT2" --daemon --non-interactive &
    mav_pid=$!

    # 포트 변경 감시자: 링크 타겟이 바뀌면 mavproxy를 죽여서 바깥 루프가
    # 새 포트로 재시작하게 함
    (
        while kill -0 "$mav_pid" 2>/dev/null; do
            if [ ! -e "$DEV" ] || [ "$(readlink -f "$DEV")" != "$target" ]; then
                log "포트 변경/연결 끊김 감지 -> mavproxy(pid $mav_pid) 재시작"
                kill "$mav_pid" 2>/dev/null
                break
            fi
            sleep 2
        done
    ) &
    watcher_pid=$!

    wait "$mav_pid" 2>/dev/null
    kill "$watcher_pid" 2>/dev/null
    log "mavproxy 종료됨 - 2초 후 재시작"
    sleep 2
done

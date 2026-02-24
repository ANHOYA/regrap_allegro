import subprocess
import time
import sys

print("🚀 [System] Teleoperation 시스템을 시작합니다.")

# 1. Isaac Sim 수신부 실행 (백그라운드가 아닌 동일한 터미널에 로그 출력)
print("⏳ [System] 1. Isaac Sim (Receiver) 환경을 구동합니다...")
receiver_process = subprocess.Popen([sys.executable, "/home/ash/projects/12_regrap_allegro/temp/receiver_isaaclab.py"])

# Isaac Sim 엔진이 켜지는 동안 잠깐 숨 고르기
time.sleep(3)

# 2. RealSense 송신부 실행
print("📷 [System] 2. RealSense Camera & MediaPipe (Sender) 구동을 시작합니다...")
sender_process = subprocess.Popen([sys.executable, "sender.py"])

try:
    # 두 프로세스가 끝날 때까지 메인 스크립트가 종료되지 않고 대기
    receiver_process.wait()
    sender_process.wait()

except KeyboardInterrupt:
    # 터미널에서 Ctrl+C 를 누르면 실행되는 종료 루틴
    print("\n🛑 [System] 사용자에 의해 시스템을 강제 종료합니다.")
    
    # 두 하위 프로세스에 종료 명령(SIGTERM) 전송
    sender_process.terminate()
    receiver_process.terminate()
    
    # 완전히 꺼질 때까지 대기
    sender_process.wait()
    receiver_process.wait()
    
    print("✅ [System] 모든 프로세스가 안전하게 종료되었습니다.")
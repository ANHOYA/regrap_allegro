# 반드시 가장 먼저 실행되어야 하는 Isaac Sim 초기화 (최신 pip 버전 방식)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import socket
import struct
import numpy as np

# SimulationApp 실행 이후에 omni 관련 모듈들을 import 해야 합니다.
from omni.isaac.core import World
from omni.isaac.core.robots import Robot
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction

# 1. UDP 수신 세팅
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
# 핵심: 데이터가 없어도 에러를 뱉고 바로 다음 코드로 넘어가도록 비동기(Non-blocking) 설정
sock.setblocking(False) 

print(f"Isaac Sim: UDP 수신 대기 중... 포트 {UDP_PORT}")

# 2. Isaac Sim World 세팅
world = World(stage_units_in_meters=1.0)
world.scene.add_default_ground_plane()

# 3. Allegro Hand 로드 (최신 5.1.0 방식)
# 🚨 [주의] 이 경로는 실제 랩실 컴퓨터에 있는 Allegro Hand의 .usd 파일 경로로 꼭 바꿔주세요!
# 예시: "file:///home/ash/projects/12_regrap_allegro/assets/allegro_hand.usd"
ALLEYGRO_HAND_USD_PATH = "omniverse://localhost/Projects/AllegroHand/allegro_hand.usd"

# Step A: USD 파일을 가상 세계의 특정 경로에 불러옵니다.
add_reference_to_stage(usd_path=ALLEYGRO_HAND_USD_PATH, prim_path="/World/AllegroHand")

# Step B: 불러온 모델을 제어 가능한 'Robot' 객체로 씌워줍니다.
allegro = world.scene.add(
    Robot(
        prim_path="/World/AllegroHand",
        name="allegro_hand",
        position=np.array([0.0, 0.0, 0.5]), # 공중에 살짝 띄워서 배치
    )
)

# 4. 시뮬레이션 초기화
world.reset()

# 16개 관절의 최신 타겟 각도를 저장할 배열 (초기값 0.0)
current_target_angles = np.zeros(16, dtype=np.float32)

# 5. 메인 시뮬레이션 및 제어 루프
try:
    while simulation_app.is_running():
        # [네트워크 수신부]
        try:
            # 64바이트(float 4byte * 16개) 데이터를 읽어옴
            data, addr = sock.recvfrom(64) 
            # C 스타일 바이트를 파이썬 튜플로 언패킹 후 numpy 배열로 변환
            received_angles = np.array(struct.unpack('16f', data), dtype=np.float32)
            
            # 노이즈 방지를 위해 간단한 Low-pass filter (선택 사항)
            # current_target_angles = 0.8 * current_target_angles + 0.2 * received_angles
            current_target_angles = received_angles
            
        except BlockingIOError:
            # 이번 시뮬레이션 스텝에 새로 도착한 UDP 패킷이 없으면 무시하고 넘어감
            pass

        # [로봇 제어부]
        # 수신된(또는 유지된) 타겟 각도를 로봇의 관절 위치 제어기(PD Controller)에 인가
        allegro.get_articulation_controller().apply_action(
            ArticulationAction(
                joint_positions=current_target_angles
            )
        )

        # 물리 엔진 1스텝 진행
        world.step(render=True)

except KeyboardInterrupt:
    print("사용자에 의해 시뮬레이션 종료")

finally:
    sock.close()
    simulation_app.close()
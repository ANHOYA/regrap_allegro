# 🚨 Isaac Sim / Isaac Lab 초기화 (가장 먼저!)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import socket
import struct
import torch
import numpy as np

# 💡 Isaac Lab 전용 모듈 임포트
import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveSceneCfg, InteractiveScene
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaacsim.storage.native import get_assets_root_path

# ---------------------------------------------------------
# 1. 문서에 기반한 커스텀 Teleop Device 클래스 (UDP -> Tensor)
# ---------------------------------------------------------
class RealSenseUDPTeleop:
    def __init__(self, port=5005, device="cuda:0"):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock.bind(("127.0.0.1", port))
        self.sock.setblocking(False)
        self.device = device
        
        # Isaac Lab은 모든 것을 GPU(PyTorch)로 계산하므로 텐서로 선언합니다.
        self.current_angles = torch.zeros((1, 16), dtype=torch.float32, device=self.device)
        print(f"[Teleop] UDP 수신 대기 중... 포트 {port}")

    def advance(self):
        """매 시뮬레이션 스텝마다 호출되어 최신 관절 각도를 Tensor로 반환"""
        try:
            data, _ = self.sock.recvfrom(64)
            angles = struct.unpack('16f', data)
            # 수신받은 값을 곧바로 GPU Tensor로 변환
            self.current_angles = torch.tensor([angles], dtype=torch.float32, device=self.device)
        except BlockingIOError:
            pass
        return self.current_angles

# ---------------------------------------------------------
# 2. Isaac Lab 환경(Scene) 세팅 클래스
# ---------------------------------------------------------
# 서버에서 모델 경로를 안전하게 가져옵니다.
assets_root_path = get_assets_root_path()
ALLEYGRO_HAND_USD_PATH = f"{assets_root_path}/Isaac/Robots/AllegroHand/allegro_hand.usd"

class AllegroSceneCfg(InteractiveSceneCfg):
    """이 클래스 하나로 바닥과 로봇 로딩, 물리 세팅이 모두 끝납니다."""
    
    # 바닥 생성
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg()
    )
    
    # Allegro Hand 생성 (Isaac Lab이 내부적으로 관절과 에러를 모두 처리해줍니다!)
    robot = ArticulationCfg(
        prim_path="/World/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ALLEYGRO_HAND_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.5), # 공중에 띄우기
        ),
    )

# ---------------------------------------------------------
# 3. 메인 실행 루프
# ---------------------------------------------------------
def main():
    # 시뮬레이션 엔진 켜기 (1/60초 고정)
    sim_cfg = sim_utils.SimulationCfg(dt=1.0/60.0, device="cuda:0")
    sim = sim_utils.SimulationContext(sim_cfg)
    
    # 씬(Scene) 생성 및 초기화
    scene_cfg = AllegroSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    
    # 통신 디바이스 켜기
    teleop = RealSenseUDPTeleop(device=sim.device)
    
    # 씬에서 로봇 객체 가져오기
    robot = scene["robot"]
    
    print("✅ [System] Isaac Lab 환경 세팅 완료! 시뮬레이션 루프를 시작합니다.")
    
    # 시뮬레이션 루프
    while simulation_app.is_running():
        # 1. 내 손가락 움직임(UDP)을 텐서(Tensor) Action으로 가져옴
        action = teleop.advance()
        
        # 2. 로봇 손에 Action(목표 각도) 인가
        robot.set_joint_position_target(action)
        
        # 3. 물리 엔진 1스텝 진행 및 렌더링 업데이트
        scene.write_data_to_sim()
        sim.step(render=True)
        scene.update(sim_cfg.dt)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[System] 사용자에 의해 종료되었습니다.")
    finally:
        simulation_app.close()
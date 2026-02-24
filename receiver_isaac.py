# 🚨 Isaac Sim 초기화 (가장 먼저!)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import sys
import socket
import struct
import numpy as np
import omni.kit.commands

def log(msg):
    """Isaac Sim이 Python stdout을 리다이렉트하므로, stderr로 출력"""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

# 💡 Isaac Sim 5.1.0 API 임포트
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.asset.importer.urdf import _urdf

# 1. UDP 수신 세팅
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

log(f"Isaac Sim: UDP 수신 대기 중... 포트 {UDP_PORT}")

# 2. Isaac Sim World 세팅 (물리 스텝을 1/60초로 고정)
world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
world.scene.add_default_ground_plane()

# 3. 로컬 URDF에서 Allegro Hand 로드
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "assets", "allegro_hand", "allegro_hand_right.urdf")

if not os.path.exists(URDF_PATH):
    raise FileNotFoundError(f"URDF 파일을 찾을 수 없습니다: {URDF_PATH}")

log(f"로컬 URDF 로딩: {URDF_PATH}")

# URDF Import 설정
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = True           # 손바닥(base_link)을 공중에 고정
import_config.make_default_prim = False
import_config.create_physics_scene = False  # World가 이미 physics scene을 만들었으므로

# URDF → USD 변환 및 Stage에 추가 (omni.kit.commands 사용)
status, prim_path = omni.kit.commands.execute(
    "URDFParseAndImportFile",
    urdf_path=URDF_PATH,
    import_config=import_config,
    dest_path="",                      # in-memory (현재 스테이지에 바로 로드)
    get_articulation_root=True,        # Articulation Root 경로를 반환받음
)

if not prim_path:
    raise Exception("URDF 임포트에 실패했습니다!")

# 스테이지 업데이트 대기
for _ in range(5):
    simulation_app.update()

log(f"✅ URDF 임포트 완료! Articulation Root: {prim_path}")

# 4. Robot 객체 생성
# 디버깅: 우선 기본 방향(손가락 위)으로 확인, 추후 물병잡기 자세로 변경 가능
# X축 180도: [0, 1, 0, 0], 기본(위): [1, 0, 0, 0]
hand_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])  # 기본 방향 (손가락 위)

allegro = world.scene.add(
    Robot(
        prim_path=prim_path,
        name="allegro_hand",
        position=np.array([0.0, 0.0, 0.3]),       # 지면에서 0.3m 위 (디버깅용)
        orientation=hand_quat_wxyz,                 # 손가락이 아래를 향하는 자세
    )
)

# 5. 시뮬레이션 초기화 (여기서 물리 엔진이 로봇을 파싱합니다)
world.reset()

# 관절 정보 출력
num_dof = allegro.num_dof
joint_names = allegro.dof_names
log(f"🤖 Allegro Hand 로드 성공! DOF 수: {num_dof}")
log(f"   관절 목록: {joint_names}")

# sender → Isaac Sim DOF 순서 매핑 테이블 구축
# sender는 joint_0 ~ joint_15를 순서대로 보냄
# Isaac Sim의 DOF 순서는 다를 수 있으므로 역매핑 필요
sender_to_sim = np.zeros(min(num_dof, 16), dtype=np.int32)
for sender_idx in range(min(num_dof, 16)):
    target_name = f"joint_{sender_idx}"
    if target_name in joint_names:
        sim_idx = list(joint_names).index(target_name)
        sender_to_sim[sender_idx] = sim_idx
    else:
        sender_to_sim[sender_idx] = sender_idx  # fallback
log(f"   매핑 테이블 (sender→sim): {sender_to_sim.tolist()}")

# 16개 관절의 최신 타겟 각도를 저장할 배열
current_target_angles = np.zeros(num_dof, dtype=np.float32)

# 6. 메인 제어 루프
try:
    while simulation_app.is_running():
        try:
            data, addr = sock.recvfrom(64)
            received_angles = np.array(struct.unpack('16f', data), dtype=np.float32)
            # sender 순서 → Isaac Sim DOF 순서로 재매핑
            for s_idx in range(min(len(received_angles), len(sender_to_sim))):
                current_target_angles[sender_to_sim[s_idx]] = received_angles[s_idx]
        except BlockingIOError:
            pass

        # 수신된 타겟 각도를 관절 위치 제어기(PD)에 인가
        allegro.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=current_target_angles)
        )

        world.step(render=True)

except KeyboardInterrupt:
    log("사용자에 의해 종료")

finally:
    sock.close()
    simulation_app.close()
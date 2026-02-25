# 🚨 Isaac Sim Initialization
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import sys
import socket
import struct
import numpy as np
import omni.kit.commands

def log(msg):
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

# 💡 Isaac Sim 5.1.0 API Import
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.asset.importer.urdf import _urdf

# 1. UDP Receive Setting
UDP_IP = "127.0.0.1"
UDP_PORT = 5005

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))
sock.setblocking(False)

log(f"Isaac Sim: UDP Receive Waiting... Port {UDP_PORT}")

# 2. Isaac Sim World Setting (Fixed Physics Step to 1/60s)
world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
world.scene.add_default_ground_plane()

# 3. Load Allegro Hand from Local URDF
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "allegro_hand_right_local.urdf")

if not os.path.exists(URDF_PATH):
    raise FileNotFoundError(f"NOT FOUND URDF: {URDF_PATH}")

log(f"Loading Local URDF: {URDF_PATH}")

# URDF Import Setting
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
# X축 180도: [0, 1, 0, 0], 기본(위): [1, 0, 0, 0]
hand_quat_wxyz = np.array([0.7071, 0.7071, 0.0, 0.0])  # 손가락 아래 방향
hand_position = np.array([0.0, 0.0, 0.7])

# fix_base=True일 때 Robot()의 position/orientation이 적용 안 될 수 있으므로
# USD API로 직접 root prim의 transform 설정
from pxr import UsdGeom, Gf
stage = omni.usd.get_context().get_stage()
# prim_path는 articulation root (예: /allegro_hand_right/root_joint)
# 상위 prim (robot 루트)에 transform 적용
robot_root_path = prim_path.rsplit("/", 1)[0]  # /allegro_hand_right
robot_prim = stage.GetPrimAtPath(robot_root_path)
if robot_prim.IsValid():
    xform = UsdGeom.Xformable(robot_prim)
    xform.ClearXformOpOrder()
    xform.AddTranslateOp().Set(Gf.Vec3d(*hand_position.tolist()))
    xform.AddOrientOp(precision=UsdGeom.XformOp.PrecisionDouble).Set(
        Gf.Quatd(float(hand_quat_wxyz[0]), float(hand_quat_wxyz[1]), float(hand_quat_wxyz[2]), float(hand_quat_wxyz[3]))
    )
    log(f"📐 Transform 설정: pos={hand_position}, quat={hand_quat_wxyz}")

allegro = world.scene.add(
    Robot(
        prim_path=prim_path,
        name="allegro_hand",
        position=hand_position,
        orientation=hand_quat_wxyz,
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
        # UDP 버퍼의 모든 패킷을 읽어서 최신 것만 사용 (지연 방지)
        latest_data = None
        while True:
            try:
                data, addr = sock.recvfrom(64)
                latest_data = data  # 계속 덮어써서 마지막(최신)만 남김
            except BlockingIOError:
                break  # 더 이상 읽을 패킷 없음

        if latest_data is not None:
            received_angles = np.array(struct.unpack('16f', latest_data), dtype=np.float32)
            # sender 순서 → Isaac Sim DOF 순서로 재매핑
            for s_idx in range(min(len(received_angles), len(sender_to_sim))):
                current_target_angles[sender_to_sim[s_idx]] = received_angles[s_idx]

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
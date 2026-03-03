# 🚨 Isaac Sim Initialization
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import sys
import socket
import struct
import time
import numpy as np
import omni.kit.commands
import omni.appwindow

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

# 3. Load Doosan A0509 + Allegro Hand Combined URDF
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "src", "doosan_allegro_combined.urdf")

if not os.path.exists(URDF_PATH):
    raise FileNotFoundError(f"NOT FOUND URDF: {URDF_PATH}")

log(f"Loading Local URDF: {URDF_PATH}")

# URDF Import Setting
import_config = _urdf.ImportConfig()
import_config.merge_fixed_joints = False
import_config.fix_base = True           # 두산 베이스를 바닥에 고정
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

# ── Configure arm joint drives (CRITICAL for position control) ──
# Doosan URDF lacks <dynamics> tags, so joints have no drive strength
# Without proper stiffness, apply_action(joint_positions=...) has ZERO effect
from pxr import UsdPhysics, PhysxSchema

stage_tmp = omni.usd.get_context().get_stage()
robot_root_path = prim_path.rsplit("/", 1)[0]  # e.g. /doosan_allegro

arm_joint_names = ["arm_joint_1", "arm_joint_2", "arm_joint_3",
                   "arm_joint_4", "arm_joint_5", "arm_joint_6"]

# Search the entire stage tree for joint prims (they're nested under parent links)
def find_joint_prim(stage, root_path, joint_name):
    """Recursively search for a joint prim by name under root_path"""
    root = stage.GetPrimAtPath(root_path)
    if not root.IsValid():
        return None
    for prim in Usd.PrimRange(root):
        if prim.GetName() == joint_name:
            return prim
    return None

from pxr import Usd
for joint_name in arm_joint_names:
    joint_prim = find_joint_prim(stage_tmp, robot_root_path, joint_name)
    if joint_prim is None:
        log(f"⚠ Joint prim not found anywhere: {joint_name}")
        continue
    
    log(f"   📍 Found joint: {joint_prim.GetPath()}")
    
    # Apply angular drive with high stiffness for position control
    drive_api = UsdPhysics.DriveAPI.Apply(joint_prim, "angular")
    drive_api.CreateTypeAttr("force")
    drive_api.CreateStiffnessAttr(1e4)   # Strong position tracking
    drive_api.CreateDampingAttr(1e3)     # Reduce oscillations  
    drive_api.CreateMaxForceAttr(300.0)  # Max torque (Nm)
    log(f"   🔧 Drive configured: {joint_name} (stiffness=1e4, damping=1e3)")

for _ in range(3):
    simulation_app.update()

# 4. Robot 객체 생성
from pxr import UsdGeom, Gf, Sdf
stage = omni.usd.get_context().get_stage()

# 통합 로봇: 두산 베이스가 바닥에 고정, 손은 arm_link_6 끝에 달림
robot_position = np.array([0.0, 0.0, 0.0])
robot_quat_wxyz = np.array([1.0, 0.0, 0.0, 0.0])

allegro = world.scene.add(
    Robot(
        prim_path=prim_path,
        name="doosan_allegro",
        position=robot_position,
        orientation=robot_quat_wxyz,
    )
)

# 5. 타겟 오브젝트 로드
from pxr import UsdPhysics, PhysxSchema, UsdShade, Usd
from target_object import get_active_object

target_cfg = get_active_object()
OBJ_PATH = target_cfg["obj_path"]
WRAPPER_USD_PATH = target_cfg["usd_path"]
TARGET_POS = Gf.Vec3d(*target_cfg["position"])

can_prim_path = "/World/target_object"
CAN_LOADED = False

if os.path.exists(OBJ_PATH):
    # OBJ를 감싸는 USD 래퍼 생성 (metersPerUnit=1.0, upAxis=Z)
    if not os.path.exists(WRAPPER_USD_PATH):
        log(f"🔄 OBJ → USD 래퍼 생성 중: {WRAPPER_USD_PATH}")
        wrapper_stage = Usd.Stage.CreateNew(WRAPPER_USD_PATH)
        UsdGeom.SetStageMetersPerUnit(wrapper_stage, 1.0)
        UsdGeom.SetStageUpAxis(wrapper_stage, UsdGeom.Tokens.z)
        root_prim = wrapper_stage.DefinePrim("/target_object", "Xform")
        root_prim.GetReferences().AddReference("./textured.obj")
        wrapper_stage.SetDefaultPrim(root_prim)
        wrapper_stage.GetRootLayer().Save()
        log(f"✅ USD 래퍼 저장 완료")
    else:
        log(f"📁 기존 USD 래퍼 사용: {WRAPPER_USD_PATH}")

    # 변환된 USD를 Isaac Sim에 로드
    from isaacsim.core.utils.stage import add_reference_to_stage
    add_reference_to_stage(WRAPPER_USD_PATH, can_prim_path)

    for _ in range(10):
        simulation_app.update()

    can_prim = stage.GetPrimAtPath(can_prim_path)
    if can_prim.IsValid():
        # 물리 속성
        UsdPhysics.RigidBodyAPI.Apply(can_prim)
        UsdPhysics.CollisionAPI.Apply(can_prim)
        mass_api = UsdPhysics.MassAPI.Apply(can_prim)
        mass_api.CreateMassAttr(target_cfg["mass"])
        CAN_LOADED = True
        # 위치 설정
        can_prim.GetAttribute("xformOp:translate").Set(TARGET_POS)
        log(f"🥫 {target_cfg['label']} loaded at {can_prim_path} pos={target_cfg['position']}")
        # 자식 prim 확인
        children = can_prim.GetChildren()
        log(f"   Children: {len(children)}")
        for c in children[:5]:
            log(f"   Child: {c.GetPath()} Type: {c.GetTypeName()}")
else:
    log(f"⚠ OBJ not found: {OBJ_PATH}")

# Fallback: OBJ도 없으면 실린더
if not CAN_LOADED:
    CAN_RADIUS = 0.033
    CAN_HEIGHT = 0.1016
    cylinder = UsdGeom.Cylinder.Define(stage, can_prim_path)
    cylinder.CreateRadiusAttr(CAN_RADIUS)
    cylinder.CreateHeightAttr(CAN_HEIGHT)
    cylinder.CreateAxisAttr("Z")
    xform = UsdGeom.Xformable(cylinder.GetPrim())
    xform.AddTranslateOp().Set(Gf.Vec3d(0.0, 0.0, CAN_HEIGHT / 2.0 + 0.01))
    mat_path = "/World/can_material"
    material = UsdShade.Material.Define(stage, mat_path)
    shader = UsdShade.Shader.Define(stage, f"{mat_path}/Shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.8, 0.1, 0.1))
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.4)
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(cylinder.GetPrim()).Bind(material)
    UsdPhysics.RigidBodyAPI.Apply(cylinder.GetPrim())
    UsdPhysics.CollisionAPI.Apply(cylinder.GetPrim())
    mass_api = UsdPhysics.MassAPI.Apply(cylinder.GetPrim())
    mass_api.CreateMassAttr(0.35)
    CAN_LOADED = True
    log(f"🥫 Cylinder fallback at {can_prim_path}")

for _ in range(5):
    simulation_app.update()

# 6. 시뮬레이션 초기화 (여기서 물리 엔진이 로봇을 파싱합니다)
world.reset()

# 관절 정보 출력
num_dof = allegro.num_dof
joint_names = allegro.dof_names
log(f"🤖 Combined Robot 로드 성공! DOF 수: {num_dof}")
log(f"   관절 목록: {list(joint_names)}")

# Arm DOF 인덱스와 Hand DOF 인덱스 분리
arm_dof_indices = []
hand_dof_indices = []
for i, name in enumerate(joint_names):
    if name.startswith("arm_"):
        arm_dof_indices.append(i)
    elif name.startswith("hand_"):
        hand_dof_indices.append(i)
log(f"   Arm DOFs: {len(arm_dof_indices)}, Hand DOFs: {len(hand_dof_indices)}")

# sender → Isaac Sim DOF 순서 매핑 테이블 구축 (hand_joint_X prefix)
sender_to_sim = np.zeros(16, dtype=np.int32)
for sender_idx in range(16):
    target_name = f"hand_joint_{sender_idx}"
    if target_name in joint_names:
        sim_idx = list(joint_names).index(target_name)
        sender_to_sim[sender_idx] = sim_idx
    else:
        log(f"   ⚠ hand_joint_{sender_idx} not found in DOF list")
        sender_to_sim[sender_idx] = 0  # fallback
log(f"   매핑 테이블 (sender→sim): {sender_to_sim.tolist()}")

# 22개 관절의 최신 타겟 각도를 저장할 배열
current_target_angles = np.zeros(num_dof, dtype=np.float32)

# Wrist target pose storage (received from sender)
wrist_target_pos = np.array([0.4, 0.0, 0.5], dtype=np.float32)  # Default: arm extended
wrist_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)  # Identity
wrist_data_valid = False

# ── IK Setup (simulation-based) ──
# Find the end-effector prim in the USD stage
ee_prim = find_joint_prim(stage, robot_root_path, "arm_link_6")
if ee_prim:
    ee_prim_path = str(ee_prim.GetPath())
    log(f"🎯 EE link found: {ee_prim_path}")
else:
    ee_prim_path = f"{robot_root_path}/arm_link_6"
    log(f"⚠ EE link not found by search, using: {ee_prim_path}")

def get_ee_world_pos():
    """Read actual end-effector world position from simulation"""
    p = stage.GetPrimAtPath(ee_prim_path)
    if not p.IsValid():
        return np.zeros(3)
    xf = UsdGeom.Xformable(p)
    T = xf.ComputeLocalToWorldTransform(0)
    t = T.ExtractTranslation()
    return np.array([t[0], t[1], t[2]], dtype=np.float64)

# Initial arm pose: bent forward, hand pointing down toward table (for grasping)
# joints: [base_yaw, shoulder, elbow, wrist_roll, wrist_pitch, wrist_yaw]
INITIAL_ARM_POSE = np.array([0.0, 0.7, 1.3, 0.0, 1.0, 0.0], dtype=np.float32)
for i, dof_idx in enumerate(arm_dof_indices):
    current_target_angles[dof_idx] = INITIAL_ARM_POSE[i]

# IK solver for arm control
from ik_solver import DoosanIK
ik_solver = DoosanIK()
_, _, ee_init = ik_solver.forward_kinematics(INITIAL_ARM_POSE)
log(f"🎯 Arm ready. Initial pose: {INITIAL_ARM_POSE.tolist()}, FK EE: {[round(x,3) for x in ee_init]}")
log(f"   Sim EE: {[round(x,3) for x in get_ee_world_pos()]}")

# ── Recording Setup ──
from recorder import DemoRecorder
import cv2

recorder = DemoRecorder(save_dir=os.path.join(SCRIPT_DIR, "demos"), image_subsample=6)

# Image reception socket (port 5006)
sock_img = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_img.bind(("127.0.0.1", 5006))
sock_img.setblocking(False)

latest_image = None  # Most recent camera frame

# Keyboard input via Isaac Sim's carb input system
import carb.input
input_iface = carb.input.acquire_input_interface()
appwindow = omni.appwindow.get_default_app_window()
keyboard = appwindow.get_keyboard()

key_pressed = {"s": False, "r": False, "t": False, "q": False}

def on_key_event(event):
    if event.type == carb.input.KeyboardEventType.KEY_PRESS:
        key_char = str(event.input).split(".")[-1].lower()
        if key_char in key_pressed:
            key_pressed[key_char] = True
    return True

keyboard_sub = input_iface.subscribe_to_keyboard_events(keyboard, on_key_event)

log("=" * 50)
log("🎮 Controls: [S]=Start  [R]=Reset  [T]=Save  [Q]=Quit")
log("=" * 50)

frame_counter = 0

# 6. Main Control Loop
try:
    while simulation_app.is_running():
        # ── Read latest joint data ──
        latest_data = None
        while True:
            try:
                data, addr = sock.recvfrom(256)
                latest_data = data
            except BlockingIOError:
                break

        if latest_data is not None:
            # Parse 23f: 16 finger + 3 wrist_pos + 4 wrist_quat
            if len(latest_data) >= 92:  # 23 * 4 bytes
                all_values = np.array(struct.unpack('23f', latest_data[:92]), dtype=np.float32)
                received_angles = all_values[:16]
                wrist_target_pos[:] = all_values[16:19]
                wrist_target_quat[:] = all_values[19:23]
                wrist_data_valid = True
            elif len(latest_data) >= 64:  # Fallback: 16f only (backward compat)
                received_angles = np.array(struct.unpack('16f', latest_data[:64]), dtype=np.float32)
            else:
                received_angles = None
            
            if received_angles is not None:
                for s_idx in range(min(len(received_angles), len(sender_to_sim))):
                    current_target_angles[sender_to_sim[s_idx]] = received_angles[s_idx]

        # ── Read latest camera image ──
        latest_img_data = None
        while True:
            try:
                img_data, _ = sock_img.recvfrom(65535)
                latest_img_data = img_data
            except BlockingIOError:
                break

        if latest_img_data is not None:
            jpg_array = np.frombuffer(latest_img_data, dtype=np.uint8)
            decoded = cv2.imdecode(jpg_array, cv2.IMREAD_COLOR)
            if decoded is not None:
                latest_image = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)

        # ── Keyboard State Machine ──
        if key_pressed["s"] and not recorder.is_recording:
            recorder.start()
            log("🔴 Recording STARTED")
            key_pressed["s"] = False

        if key_pressed["r"] and recorder.is_recording:
            recorder.discard()
            # 타겟 오브젝트 원위치 복원
            if CAN_LOADED:
                try:
                    can_prim.GetAttribute("xformOp:translate").Set(TARGET_POS)
                    orient_attr = can_prim.GetAttribute("xformOp:orient")
                    if orient_attr:
                        try:
                            orient_attr.Set(Gf.Quatf(1, 0, 0, 0))
                        except Exception:
                            orient_attr.Set(Gf.Quatd(1, 0, 0, 0))
                    # 물리 속도 초기화
                    vel = can_prim.GetAttribute("physics:velocity")
                    if vel:
                        vel.Set(Gf.Vec3f(0, 0, 0))
                    ang = can_prim.GetAttribute("physics:angularVelocity")
                    if ang:
                        ang.Set(Gf.Vec3f(0, 0, 0))
                except Exception as e:
                    log(f"⚠ Reset error: {e}")
            log("⏹ Recording RESET (data discarded). Waiting 1s...")
            time.sleep(1)
            log("🎮 Ready. Press [S] to start recording.")
            key_pressed["r"] = False

        if key_pressed["t"] and recorder.is_recording:
            saved_path = recorder.save(joint_names=list(joint_names))
            if saved_path:
                log(f"💾 Saved → {saved_path}")
            else:
                log("⚠ Nothing to save")
            # 타겟 오브젝트 원위치 복원
            if CAN_LOADED:
                try:
                    can_prim.GetAttribute("xformOp:translate").Set(TARGET_POS)
                    orient_attr = can_prim.GetAttribute("xformOp:orient")
                    if orient_attr:
                        try:
                            orient_attr.Set(Gf.Quatf(1, 0, 0, 0))
                        except Exception:
                            orient_attr.Set(Gf.Quatd(1, 0, 0, 0))
                    vel = can_prim.GetAttribute("physics:velocity")
                    if vel:
                        vel.Set(Gf.Vec3f(0, 0, 0))
                    ang = can_prim.GetAttribute("physics:angularVelocity")
                    if ang:
                        ang.Set(Gf.Vec3f(0, 0, 0))
                except Exception as e:
                    log(f"⚠ Reset error: {e}")
            log("Waiting 1s...")
            time.sleep(1)
            log("🎮 Ready. Press [S] to start recording.")
            key_pressed["t"] = False

        if key_pressed["q"]:
            if recorder.is_recording:
                recorder.discard()
            log("👋 Quit requested")
            key_pressed["q"] = False
            break

        # ── Record frame ──
        if recorder.is_recording:
            recorder.add_frame(current_target_angles, latest_image)

        # ── IK for arm joints (analytical FK + simulation feedback) ──
        if wrist_data_valid:
            # Get current arm joint angles
            q_arm = np.array([current_target_angles[idx] for idx in arm_dof_indices], dtype=np.float32)
            
            # Use IK solver: 2 iterations per frame for smooth tracking
            q_new = ik_solver.solve_ik(q_arm, wrist_target_pos, max_iter=2, gain=0.8, damping=0.1)
            
            # Smooth interpolation to prevent jerky motion
            alpha = 0.3  # Blend factor (0=no change, 1=full IK)
            q_blended = q_arm * (1 - alpha) + q_new * alpha
            
            for i, dof_idx in enumerate(arm_dof_indices):
                current_target_angles[dof_idx] = q_blended[i]

        # ── Apply joint positions ──
        allegro.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=current_target_angles)
        )

        world.step(render=True)

        # ── Status display (every 60 frames) ──
        frame_counter += 1
        if frame_counter % 60 == 0:
            log(recorder.status_str())
            if wrist_data_valid:
                ee_pos = get_ee_world_pos()
                log(f"   🎯 target={[round(float(x),3) for x in wrist_target_pos]}")
                log(f"   📍 actual_ee={[round(x,3) for x in ee_pos]}")
                log(f"   📐 arm_q={[round(float(current_target_angles[idx]),2) for idx in arm_dof_indices]}")

except KeyboardInterrupt:
    if recorder.is_recording:
        recorder.discard()
    log("Interrupted by user")

finally:
    input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    sock.close()
    sock_img.close()
    simulation_app.close()
# 🚨 Isaac Sim Initialization
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})

import os
import sys
import time
import numpy as np
import omni.kit.commands
import omni.appwindow

def log(msg):
    """Log to stderr, safely handling Unicode that crashes Isaac Sim's stderr handler"""
    try:
        safe_msg = msg.encode('ascii', errors='replace').decode('ascii')
        sys.stderr.write(safe_msg + "\n")
        sys.stderr.flush()
    except Exception:
        try:
            print(msg)
        except Exception:
            pass

# Isaac Sim 5.1.0 API Import
from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.asset.importer.urdf import _urdf

# ============================================================
# Vision Pro Configuration
# ============================================================
VISIONPRO_IP = "192.168.0.36"  # <-- Change this to your Vision Pro IP

from avp_stream import VisionProStreamer
vp = VisionProStreamer(ip=VISIONPRO_IP)
log(f"[VP] VisionProStreamer connecting to {VISIONPRO_IP}...")

# Coordinate frame transform: Vision Pro -> Robot (from keti_retargeting)
OPERATOR2VP_RIGHT = np.array([
    [0, 0, -1],
    [-1, 0, 0],
    [0, 1, 0],
], dtype=np.float64)

# Rx(180deg) quaternion in [w,x,y,z] format: cos(90)=0, sin(90)=1 → [0,1,0,0]
RX_PI_WXYZ = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)

def rot_matrix_to_quat_wxyz(R):
    """Convert 3x3 rotation matrix to quaternion [w,x,y,z]."""
    tr = R[0,0] + R[1,1] + R[2,2]
    if tr > 0:
        s = 0.5 / np.sqrt(tr + 1.0)
        w = 0.25 / s
        x = (R[2,1] - R[1,2]) * s
        y = (R[0,2] - R[2,0]) * s
        z = (R[1,0] - R[0,1]) * s
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        w = (R[2,1] - R[1,2]) / s
        x = 0.25 * s
        y = (R[0,1] + R[1,0]) / s
        z = (R[0,2] + R[2,0]) / s
    elif R[1,1] > R[2,2]:
        s = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        w = (R[0,2] - R[2,0]) / s
        x = (R[0,1] + R[1,0]) / s
        y = 0.25 * s
        z = (R[1,2] + R[2,1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        w = (R[1,0] - R[0,1]) / s
        x = (R[0,2] + R[2,0]) / s
        y = (R[1,2] + R[2,1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / (np.linalg.norm(q) + 1e-8)

def quat_multiply_wxyz(q1, q2):
    """Hamilton product q1*q2, both in [w,x,y,z] format."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=np.float64)

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
        # Apply friction material
        mat_path = "/World/target_physics_material"
        physics_mat = UsdShade.Material.Define(stage, mat_path)
        phys_api = UsdPhysics.MaterialAPI.Apply(physics_mat.GetPrim())
        phys_api.CreateStaticFrictionAttr(target_cfg.get("static_friction", 0.8))
        phys_api.CreateDynamicFrictionAttr(target_cfg.get("dynamic_friction", 0.6))
        phys_api.CreateRestitutionAttr(target_cfg.get("restitution", 0.1))
        UsdShade.MaterialBindingAPI(can_prim).Bind(physics_mat, UsdShade.Tokens.weakerThanDescendants, "physics")
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

# Wrist target pose storage (from Vision Pro)
wrist_target_pos = np.array([0.0, 0.5, 0.5], dtype=np.float32)  # Will be overwritten after IK init
wrist_target_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)  # Identity [w,x,y,z]
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

# -- Isaac Sim Built-in IK Solver (LulaKinematicsSolver) --
from isaacsim.robot_motion.motion_generation import (
    LulaKinematicsSolver,
    ArticulationKinematicsSolver,
)

ROBOT_DESC_PATH = os.path.join(SCRIPT_DIR, "src", "doosan_allegro_robot_descriptor.yaml")
URDF_FOR_IK = os.path.join(SCRIPT_DIR, "src", "doosan_allegro_combined.urdf")

# Target EE position for initial spawn
INITIAL_EE_TARGET = np.array([0.0, 0.5, 0.3], dtype=np.float64)

# ── Workspace Safety ──
Z_MIN_LIMIT = 0.05       # EE target Z minimum (5cm above ground)
Z_MIN_ELBOW = 0.08       # Elbow Z minimum (8cm above ground)
WORKSPACE_RADIUS = 0.5   # Max delta from initial position

# Store last known safe arm configuration
previous_good_arm_q = None

try:
    lula_kinematics = LulaKinematicsSolver(
        robot_description_path=ROBOT_DESC_PATH,
        urdf_path=URDF_FOR_IK,
    )
    art_kinematics = ArticulationKinematicsSolver(
        robot_articulation=allegro,
        kinematics_solver=lula_kinematics,
        end_effector_frame_name="arm_link_6",
    )
    IK_READY = True
    log("[OK] Lula IK solver ready! EE frame: arm_link_6")
    
    # Solve IK for initial EE target (position-only, no orientation constraint)
    init_action, init_success = art_kinematics.compute_inverse_kinematics(
        target_position=INITIAL_EE_TARGET,
        target_orientation=None,
    )
    if init_success and init_action is not None and init_action.joint_positions is not None:
        for i, dof_idx in enumerate(arm_dof_indices):
            val = init_action.joint_positions[dof_idx]
            if val is not None and not np.isnan(val):
                current_target_angles[dof_idx] = float(val)
        log(f"[OK] IK initial pose solved for EE={INITIAL_EE_TARGET.tolist()}")
        log(f"   arm_q={[round(float(current_target_angles[idx]),3) for idx in arm_dof_indices]}")
    else:
        log("[WARN] IK failed for initial pose, using fallback joint angles")
        FALLBACK_POSE = [0.0, 0.5, -1.3, 0.0, 1.0, 0.0]
        for i, dof_idx in enumerate(arm_dof_indices):
            current_target_angles[dof_idx] = FALLBACK_POSE[i]

except Exception as e:
    IK_READY = False
    log(f"[WARN] Lula IK failed to init: {e}")
    log(f"   Falling back to fixed arm pose.")
    FALLBACK_POSE = [0.0, 0.5, -1.3, 0.0, 1.0, 0.0]
    for i, dof_idx in enumerate(arm_dof_indices):
        current_target_angles[dof_idx] = FALLBACK_POSE[i]

# -- First-seen calibration state --
# When hand first appears, record that camera position as the reference.
# All subsequent movements are RELATIVE to this reference, mapped around INITIAL_EE_TARGET.
calib_reference = None  # Will be set on first valid wrist data
wrist_target_pos = INITIAL_EE_TARGET.copy().astype(np.float32)

log(f"[INFO] Arm ready. Sim EE: {[round(x,3) for x in get_ee_world_pos()]}")
log(f"   Calibration: first-seen hand position = EE {INITIAL_EE_TARGET.tolist()}")

# -- Set viewport camera: position [0,0,1], looking toward [0,1,0] --
try:
    from pxr import UsdGeom, Gf
    # Create a dedicated camera prim
    cam_path = "/World/teleop_camera"
    cam_prim = stage.GetPrimAtPath(cam_path)
    if not cam_prim.IsValid():
        cam_prim = UsdGeom.Camera.Define(stage, cam_path).GetPrim()
    xformable = UsdGeom.Xformable(cam_prim)
    xformable.ClearXformOpOrder()
    # Camera transform (user-tuned)
    translate_op = xformable.AddTranslateOp()
    translate_op.Set(Gf.Vec3d(0.0, -0.7, 2.6))
    rotate_op = xformable.AddRotateXYZOp()
    rotate_op.Set(Gf.Vec3d(26, -0.0, 0))
    # Set focal length 24mm
    cam_api = UsdGeom.Camera(cam_prim)
    cam_api.GetFocalLengthAttr().Set(24.0)
    
    for _ in range(3):
        simulation_app.update()
    
    # Set viewport to use this camera
    from omni.kit.viewport.utility import get_active_viewport
    viewport_api = get_active_viewport()
    if viewport_api:
        viewport_api.camera_path = cam_path
        log(f"[OK] Viewport camera set to {cam_path}")
    else:
        log("[WARN] Could not get active viewport")
except Exception as e:
    log(f"[WARN] Viewport camera setup failed: {e}")

# ── Recording Setup ──
from recorder import DemoRecorder

recorder = DemoRecorder(save_dir=os.path.join(SCRIPT_DIR, "demos"), image_subsample=6)

latest_image = None  # No camera image from VP (can add later)

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

# -- VP finger joint indices for retargeting --
# VP skeleton: [0]wrist, [1-4]thumb, [5-9]index, [10-14]middle, [15-19]ring, [20-24]little
# We compute finger curl angles from joint positions to map to Allegro
def vp_fingers_to_allegro(finger_transforms):
    """Convert VP right hand finger transforms (25x4x4) to Allegro 16 joint angles.
    
    Uses joint position vectors to compute curl angles for each finger.
    VP joints per finger: [metacarpal, knuckle, intermediateBase, intermediateTip, tip]
    Allegro joints per finger: [abduction, MCP_flex, PIP_flex, DIP_flex]
    """
    def angle_between(v1, v2):
        c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
        return np.arccos(np.clip(c, -1, 1))
    
    angles = np.zeros(16, dtype=np.float32)
    
    # Extract 3D positions from 4x4 transforms
    pos = finger_transforms[:, :3, 3]  # (25, 3)
    
    # Index finger (VP: 5-9, Allegro: joints 0-3)
    idx_vecs = [pos[6]-pos[5], pos[7]-pos[6], pos[8]-pos[7], pos[9]-pos[8]]
    angles[0] = 0.0  # index abduction (TODO: compute from palm plane)
    angles[1] = angle_between(idx_vecs[0], idx_vecs[1])  # MCP
    angles[2] = angle_between(idx_vecs[1], idx_vecs[2])  # PIP
    angles[3] = angle_between(idx_vecs[2], idx_vecs[3])  # DIP
    
    # Middle finger (VP: 10-14, Allegro: joints 4-7)
    mid_vecs = [pos[11]-pos[10], pos[12]-pos[11], pos[13]-pos[12], pos[14]-pos[13]]
    angles[4] = 0.0  # abduction
    angles[5] = angle_between(mid_vecs[0], mid_vecs[1])
    angles[6] = angle_between(mid_vecs[1], mid_vecs[2])
    angles[7] = angle_between(mid_vecs[2], mid_vecs[3])
    
    # Ring finger (VP: 15-19, Allegro: joints 8-11)
    ring_vecs = [pos[16]-pos[15], pos[17]-pos[16], pos[18]-pos[17], pos[19]-pos[18]]
    angles[8] = 0.0  # abduction
    angles[9] = angle_between(ring_vecs[0], ring_vecs[1])
    angles[10] = angle_between(ring_vecs[1], ring_vecs[2])
    angles[11] = angle_between(ring_vecs[2], ring_vecs[3])
    
    # Thumb (VP: 1-4, Allegro: joints 12-15)
    # VP thumb joints: 0=wrist, 1=thumbCMC, 2=thumbMCP, 3=thumbIP, 4=thumbTip
    th_vecs = [pos[1]-pos[0], pos[2]-pos[1], pos[3]-pos[2], pos[4]-pos[3]]
    
    # Joint 12: Thumb rotation/abduction (CMC) - use palm-to-thumb angle
    # Compute palm plane from index and pinky metacarpals
    palm_y = pos[5] - pos[0]   # wrist -> index metacarpal
    palm_x = pos[15] - pos[0]  # wrist -> ring metacarpal
    palm_normal = np.cross(palm_y, palm_x)
    palm_normal = palm_normal / (np.linalg.norm(palm_normal) + 1e-8)
    thumb_dir = pos[2] - pos[1]  # metacarpal -> knuckle
    thumb_dir = thumb_dir / (np.linalg.norm(thumb_dir) + 1e-8)
    # Angle between palm normal and thumb direction
    dot_val = np.clip(np.dot(palm_normal, thumb_dir), -1, 1)
    thumb_rotation = np.abs(np.arcsin(dot_val))
    angles[12] = thumb_rotation * 1.5  # Gain for better range
    
    # Joint 13: MCP flexion (amplified)
    angles[13] = angle_between(th_vecs[0], th_vecs[1]) * 1.3
    # Joint 14: IP flexion (amplified)
    angles[14] = angle_between(th_vecs[1], th_vecs[2]) * 1.3
    # Joint 15: DIP coupled
    angles[15] = angle_between(th_vecs[2], th_vecs[3]) * 0.8
    
    return angles

log("=" * 50)
log("Controls: [S]=Start  [R]=Reset  [T]=Save  [Q]=Quit")
log("=" * 50)

frame_counter = 0

# 6. Main Control Loop
try:
    while simulation_app.is_running():
        # -- Read hand tracking from Vision Pro --
        vp_data = vp.get_latest()
        
        if vp_data is not None:
            try:
                # Get right wrist SE3 transform (4x4)
                rw = np.array(vp_data["right_wrist"][0], dtype=np.float64).copy()
                
                # Apply coordinate frame transform to rotation (VP -> Robot)
                rw[:3, :3] = rw[:3, :3] @ OPERATOR2VP_RIGHT
                raw_wrist_pos = rw[:3, 3].copy()  # 3D position in meters
                
                # -- Wrist ORIENTATION (following keti_retargeting) --
                # 1. Extract quaternion from transformed rotation
                quat_wxyz = rot_matrix_to_quat_wxyz(rw[:3, :3])
                # 2. Use directly (OPERATOR2VP_RIGHT handles frame; no Rx(pi) needed for our setup)
                target_quat = quat_wxyz / (np.linalg.norm(quat_wxyz) + 1e-8)
                # 3. Smooth orientation (SLERP approximation via averaging)
                quat_smooth = 0.3
                # Ensure quaternion hemisphere consistency (avoid sign flip)
                if np.dot(wrist_target_quat, target_quat) < 0:
                    target_quat = -target_quat
                wrist_target_quat[:] = wrist_target_quat * (1 - quat_smooth) + target_quat * quat_smooth
                wrist_target_quat[:] = wrist_target_quat / (np.linalg.norm(wrist_target_quat) + 1e-8)
                
                # -- Wrist POSITION --
                # First-seen calibration
                if calib_reference is None:
                    calib_reference = raw_wrist_pos.copy()
                    log(f"[CALIB] First-seen VP wrist: {[round(x,3) for x in calib_reference]}")
                    log(f"   Maps to robot EE = {INITIAL_EE_TARGET.tolist()}")
                
                # VP -> Isaac Sim coordinate mapping (direct 1:1)
                vp_delta = raw_wrist_pos - calib_reference
                robot_pos = INITIAL_EE_TARGET + vp_delta
                
                # Clip to safe workspace
                robot_pos[0] = np.clip(robot_pos[0], INITIAL_EE_TARGET[0] - WORKSPACE_RADIUS, INITIAL_EE_TARGET[0] + WORKSPACE_RADIUS)
                robot_pos[1] = np.clip(robot_pos[1], INITIAL_EE_TARGET[1] - WORKSPACE_RADIUS, INITIAL_EE_TARGET[1] + WORKSPACE_RADIUS)
                # Z-axis: CRITICAL - enforce minimum height to prevent ground penetration
                robot_pos[2] = np.clip(robot_pos[2], Z_MIN_LIMIT, 0.9)
                
                # Low-pass filter
                smoothing = 0.3
                wrist_target_pos[:] = wrist_target_pos * (1 - smoothing) + robot_pos.astype(np.float32) * smoothing
                wrist_data_valid = True
                
                # -- Finger retargeting --
                right_fingers = np.array(vp_data["right_fingers"], dtype=np.float64)  # (25, 4, 4)
                finger_angles = vp_fingers_to_allegro(right_fingers)
                
                # Map finger angles to Allegro DOFs
                for s_idx in range(min(len(finger_angles), len(sender_to_sim))):
                    current_target_angles[sender_to_sim[s_idx]] = finger_angles[s_idx]
                    
            except Exception as e:
                pass  # Skip frame on VP data errors

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

        # -- IK for arm joints (Isaac Sim LulaKinematicsSolver) --
        if wrist_data_valid and IK_READY:
            # Position + Orientation IK (following keti_retargeting pipeline)
            ik_action, ik_success = art_kinematics.compute_inverse_kinematics(
                target_position=wrist_target_pos.astype(np.float64),
                target_orientation=wrist_target_quat,
            )
            
            if ik_success and ik_action is not None:
                ik_positions = ik_action.joint_positions
                if ik_positions is not None:
                    # -- Post-IK Safety: check elbow height via FK --
                    # Get arm_link_3 (elbow) world position to check Z
                    ik_safe = True
                    try:
                        elbow_prim = stage.GetPrimAtPath("/World/doosan_allegro/arm_link_3")
                        if elbow_prim.IsValid():
                            from pxr import UsdGeom
                            xformable = UsdGeom.Xformable(elbow_prim)
                            world_tf = xformable.ComputeLocalToWorldTransform(0)
                            elbow_z = world_tf.GetRow(3)[2]
                            if elbow_z < Z_MIN_ELBOW:
                                ik_safe = False  # Reject: elbow below ground
                    except Exception:
                        pass  # If FK check fails, accept the IK solution
                    
                    if ik_safe:
                        alpha = 0.3  # Smooth blending factor
                        for i, dof_idx in enumerate(arm_dof_indices):
                            if ik_positions[dof_idx] is not None and not np.isnan(ik_positions[dof_idx]):
                                current_target_angles[dof_idx] = (
                                    current_target_angles[dof_idx] * (1 - alpha) +
                                    float(ik_positions[dof_idx]) * alpha
                                )
                        # Store as last known good configuration
                        previous_good_arm_q = [float(current_target_angles[idx]) for idx in arm_dof_indices]
                    else:
                        # Elbow too low - keep previous safe configuration
                        if previous_good_arm_q is not None:
                            for i, dof_idx in enumerate(arm_dof_indices):
                                current_target_angles[dof_idx] = previous_good_arm_q[i]

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
except Exception as e:
    import traceback
    log(f"FATAL ERROR in main loop: {e}")
    log(traceback.format_exc())

finally:
    input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    simulation_app.close()
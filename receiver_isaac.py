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

# 3. Load Allegro Hand from Local URDF
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "src", "allegro", "allegro_hand_right_local.urdf")

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
hand_position = np.array([0.0, 0.0, 0.1])

# fix_base=True일 때 Robot()의 position/orientation이 적용 안 될 수 있으므로
# USD API로 직접 root prim의 transform 설정
from pxr import UsdGeom, Gf, Sdf
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
                data, addr = sock.recvfrom(64)
                latest_data = data
            except BlockingIOError:
                break

        if latest_data is not None:
            received_angles = np.array(struct.unpack('16f', latest_data), dtype=np.float32)
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

        # ── Apply joint positions ──
        allegro.get_articulation_controller().apply_action(
            ArticulationAction(joint_positions=current_target_angles)
        )

        world.step(render=True)

        # ── Status display (every 60 frames) ──
        frame_counter += 1
        if frame_counter % 60 == 0:
            log(recorder.status_str())

except KeyboardInterrupt:
    if recorder.is_recording:
        recorder.discard()
    log("Interrupted by user")

finally:
    input_iface.unsubscribe_to_keyboard_events(keyboard, keyboard_sub)
    sock.close()
    sock_img.close()
    simulation_app.close()
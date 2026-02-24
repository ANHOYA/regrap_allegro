# 🔍 USD Stage 디버그 스크립트
# Allegro Hand USD를 로드하고 전체 Prim 트리를 출력하여
# Articulation Root의 정확한 경로를 찾아냅니다.

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})  # GUI 없이 빠르게

import time
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdPhysics, UsdGeom, Usd

# 1. Nucleus 에셋 경로
assets_root_path = get_assets_root_path()
if assets_root_path is None:
    print("❌ NVIDIA Nucleus 서버에 연결할 수 없습니다.")
    simulation_app.close()
    exit()

print(f"✅ Assets root path: {assets_root_path}")

ALLEGRO_HAND_USD_PATH = f"{assets_root_path}/Isaac/Robots/AllegroHand/allegro_hand.usd"
BASE_PRIM_PATH = "/World/AllegroHand"

print(f"📦 USD 로드 중: {ALLEGRO_HAND_USD_PATH}")
add_reference_to_stage(usd_path=ALLEGRO_HAND_USD_PATH, prim_path=BASE_PRIM_PATH)

# 여러 번 update 하여 스테이지 완전히 로드 대기
for i in range(10):
    simulation_app.update()
time.sleep(2)
for i in range(5):
    simulation_app.update()

# 2. 전체 Stage 트리 출력
stage = get_current_stage()
print("\n" + "=" * 80)
print("📋 전체 Prim 트리 (/ 이하)")
print("=" * 80)

articulation_roots = []
joint_prims = []

for prim in stage.TraverseAll():
    path = prim.GetPath().pathString
    prim_type = prim.GetTypeName()
    indent = "  " * (path.count("/") - 1)
    
    # API 정보 수집
    apis = []
    if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        apis.append("🎯 ArticulationRootAPI")
        articulation_roots.append(path)
    if prim.HasAPI(UsdPhysics.RigidBodyAPI):
        apis.append("RigidBodyAPI")
    if prim.HasAPI(UsdPhysics.CollisionAPI):
        apis.append("CollisionAPI")
    
    # Joint 타입인지 체크 (type name 기반)
    if "Joint" in prim_type:
        joint_prims.append(path)
        apis.append(f"Joint({prim_type})")
    
    api_str = f"  [{', '.join(apis)}]" if apis else ""
    
    # /World/AllegroHand 아래만 상세 출력
    if path.startswith(BASE_PRIM_PATH) or path in ("/", "/World"):
        print(f"{indent}{path}  (type: {prim_type}){api_str}")

print("\n" + "=" * 80)
print("🎯 발견된 Articulation Root 목록:")
print("=" * 80)
if articulation_roots:
    for ar in articulation_roots:
        print(f"  ✅ {ar}")
else:
    print("  ❌ ArticulationRootAPI를 가진 Prim이 없습니다!")
    print("     → PhysX Articulation이 implicit으로 정의되어 있을 수 있습니다.")
    print(f"     → BASE_PRIM_PATH ({BASE_PRIM_PATH})를 직접 사용해보세요.")

print(f"\n🔗 발견된 Joint Prim 수: {len(joint_prims)}")
if joint_prims:
    for j in joint_prims[:5]:
        print(f"  - {j}")
    if len(joint_prims) > 5:
        print(f"  ... 외 {len(joint_prims) - 5}개")

# 3. BASE_PRIM_PATH의 직접 자식들 나열
print(f"\n📂 {BASE_PRIM_PATH} 의 직접 자식:")
base_prim = stage.GetPrimAtPath(BASE_PRIM_PATH)
if base_prim.IsValid():
    for child in base_prim.GetChildren():
        child_type = child.GetTypeName()
        apis = []
        if child.HasAPI(UsdPhysics.ArticulationRootAPI):
            apis.append("🎯 ArticulationRootAPI")
        if child.HasAPI(UsdPhysics.RigidBodyAPI):
            apis.append("RigidBodyAPI")
        api_str = f"  [{', '.join(apis)}]" if apis else ""
        print(f"  - {child.GetPath().pathString}  (type: {child_type}){api_str}")
else:
    print(f"  ❌ {BASE_PRIM_PATH} Prim이 존재하지 않습니다!")

simulation_app.close()
print("\n✅ 디버그 완료")

# 🔍 디버그 스크립트 v2: 에셋 다운로드 대기 + 전체 Prim 덤프
# Nucleus S3에서 USD를 가져올 때 비동기로 다운로드되므로
# 충분히 대기한 후 Prim 트리를 확인합니다.

from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import time
from isaacsim.core.utils.stage import add_reference_to_stage, get_current_stage
from isaacsim.storage.native import get_assets_root_path
from pxr import UsdPhysics, Usd

assets_root_path = get_assets_root_path()
print(f"Assets root: {assets_root_path}")

ALLEGRO_USD = f"{assets_root_path}/Isaac/Robots/AllegroHand/allegro_hand.usd"
BASE_PRIM = "/World/AllegroHand"

print(f"Loading: {ALLEGRO_USD}")
add_reference_to_stage(usd_path=ALLEGRO_USD, prim_path=BASE_PRIM)

# 충분히 기다림 - 원격 USD 다운로드 대기
print("Waiting for USD to fully resolve (30 updates + 5s sleep)...")
for i in range(30):
    simulation_app.update()
    time.sleep(0.2)

time.sleep(5)

for i in range(20):
    simulation_app.update()

stage = get_current_stage()

# 전체 Prim 덤프
print("\n" + "=" * 80)
count = 0
for prim in stage.TraverseAll():
    path = prim.GetPath().pathString
    if path.startswith(BASE_PRIM) or path in ("/", "/World"):
        ptype = prim.GetTypeName()
        apis = []
        if prim.HasAPI(UsdPhysics.ArticulationRootAPI):
            apis.append("🎯ART_ROOT")
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            apis.append("RIGID")
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            apis.append("COLL")
        if "Joint" in ptype:
            apis.append(f"JOINT")
        
        depth = path.count("/") - 1
        indent = "  " * depth
        api_str = f" [{','.join(apis)}]" if apis else ""
        print(f"{indent}{path} ({ptype}){api_str}")
        count += 1

print(f"\nTotal prims under {BASE_PRIM}: {count}")

# 자식 직접 확인
base = stage.GetPrimAtPath(BASE_PRIM)
if base.IsValid():
    children = list(base.GetChildren())
    print(f"Direct children of {BASE_PRIM}: {len(children)}")
    for c in children:
        print(f"  - {c.GetPath()} ({c.GetTypeName()})")
else:
    print(f"❌ {BASE_PRIM} is not valid!")

simulation_app.close()

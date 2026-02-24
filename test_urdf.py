# 🔍 URDF Import 테스트 v2 (결과를 파일에 기록)
from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": True})

import os
import sys
import numpy as np
import omni.kit.commands

from isaacsim.core.api import World
from isaacsim.core.api.robots import Robot
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.asset.importer.urdf import _urdf

RESULT_FILE = "/tmp/urdf_test_result.txt"

def log(msg):
    """Isaac Sim에 의해 stdout이 리다이렉트되므로, 파일에 직접 기록"""
    with open(RESULT_FILE, "a") as f:
        f.write(msg + "\n")
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()

# 결과파일 초기화
open(RESULT_FILE, "w").close()

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
URDF_PATH = os.path.join(SCRIPT_DIR, "assets", "allegro_hand", "allegro_hand_right.urdf")

log(f"URDF path: {URDF_PATH}")
log(f"File exists: {os.path.exists(URDF_PATH)}")

try:
    # World 생성
    world = World(stage_units_in_meters=1.0, physics_dt=1.0/60.0)
    world.scene.add_default_ground_plane()
    log("World created")

    # URDF Import Config
    import_config = _urdf.ImportConfig()
    import_config.merge_fixed_joints = False
    import_config.fix_base = True
    import_config.make_default_prim = False
    import_config.create_physics_scene = False

    log("Calling URDFParseAndImportFile...")

    status, prim_path = omni.kit.commands.execute(
        "URDFParseAndImportFile",
        urdf_path=URDF_PATH,
        import_config=import_config,
        dest_path="",
        get_articulation_root=True,
    )

    log(f"status={status}, prim_path={prim_path}")

    for _ in range(5):
        simulation_app.update()

    if prim_path:
        allegro = world.scene.add(
            Robot(prim_path=prim_path, name="allegro_hand", position=np.array([0, 0, 0.5]))
        )
        world.reset()
        log(f"SUCCESS! DOF: {allegro.num_dof}")
        log(f"Joint names: {allegro.dof_names}")
    else:
        log("FAILED: URDF import returned no prim_path!")

except Exception as e:
    log(f"EXCEPTION: {type(e).__name__}: {e}")
    import traceback
    log(traceback.format_exc())

finally:
    simulation_app.close()
    log("Done!")

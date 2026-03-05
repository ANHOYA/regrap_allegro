"""
YCB Target Object Configuration Module
To switch objects, simply change ACTIVE_OBJECT.
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════
#  YCB Object Catalog
#  To add a new object: add entry to dict, then change ACTIVE_OBJECT
# ═══════════════════════════════════════════
YCB_OBJECTS = {
    "master_chef_can": {
        "folder": "002_master_chef_can",
        "mass": 0.414,
        # "position": (0.08, -0.12, 0.0),
        "position": (0.08, 0.55, 0.0),
        "label": "Master Chef Can",
        "static_friction": 0.8,
        "dynamic_friction": 0.6,
        "restitution": 0.1,
    },
    "tomato_soup_can": {
        "folder": "005_tomato_soup_can",
        "mass": 0.35,
        "position": (0.05, 0.5, 0.0),
        "label": "Tomato Soup Can",
        "static_friction": 0.8,
        "dynamic_friction": 0.6,
        "restitution": 0.1,
    },
}

# ═══════════════════════════════════════════
#  ★ Change this to switch objects ★
# ═══════════════════════════════════════════
ACTIVE_OBJECT = "master_chef_can"


def get_active_object():
    """Return config for the active object (including resolved paths)"""
    if ACTIVE_OBJECT not in YCB_OBJECTS:
        raise ValueError(f"Unknown object: '{ACTIVE_OBJECT}'. Available: {list(YCB_OBJECTS.keys())}")
    
    obj = YCB_OBJECTS[ACTIVE_OBJECT].copy()
    base_dir = os.path.join(SCRIPT_DIR, "assets", "ycb", obj["folder"], "google_16k")
    obj["obj_path"] = os.path.join(base_dir, "textured.obj")
    obj["usd_path"] = os.path.join(base_dir, "textured_converted.usd")
    return obj
"""
YCB 타겟 오브젝트 설정 모듈
물체를 교체하려면 ACTIVE_OBJECT만 변경하세요.
"""
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ═══════════════════════════════════════════
#  YCB 물체 카탈로그
#  새 물체 추가: 딕셔너리에 항목 추가 후 ACTIVE_OBJECT 변경
# ═══════════════════════════════════════════
YCB_OBJECTS = {
    "master_chef_can": {
        "folder": "002_master_chef_can",
        "mass": 0.414,
        "position": (0.08, -0.12, 0.0),
        "label": "Master Chef Can",
    },
    "tomato_soup_can": {
        "folder": "005_tomato_soup_can",
        "mass": 0.35,
        "position": (0.08, -0.12, 0.0),
        "label": "Tomato Soup Can",
    },
}

# ═══════════════════════════════════════════
#  ★ 여기만 바꾸면 물체 교체 ★
# ═══════════════════════════════════════════
ACTIVE_OBJECT = "master_chef_can"


def get_active_object():
    """현재 활성 물체의 설정값 반환 (경로 포함)"""
    if ACTIVE_OBJECT not in YCB_OBJECTS:
        raise ValueError(f"Unknown object: '{ACTIVE_OBJECT}'. Available: {list(YCB_OBJECTS.keys())}")
    
    obj = YCB_OBJECTS[ACTIVE_OBJECT].copy()
    base_dir = os.path.join(SCRIPT_DIR, "assets", "ycb", obj["folder"], "google_16k")
    obj["obj_path"] = os.path.join(base_dir, "textured.obj")
    obj["usd_path"] = os.path.join(base_dir, "textured_converted.usd")
    return obj

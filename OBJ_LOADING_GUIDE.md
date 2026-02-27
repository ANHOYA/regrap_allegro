# Isaac Sim에서 YCB OBJ 모델 로딩 가이드

## 문제

Isaac Sim은 OBJ 파일을 직접 참조(`CreateReference`)하면 **USD Metrics Assembler**가 자동으로 단위/축 변환을 적용합니다:

- `ScaleUnitsResolve = 0.01` (cm → m 변환)
- `RotateUnitsResolve = 90` (Y-up → Z-up 변환)

이로 인해 물체가 **100배 작게** 보이고 **90도 회전**되는 문제가 발생합니다.

## 해결 방법: USD 래퍼

OBJ를 직접 참조하지 않고, **올바른 stage metadata를 가진 USD 래퍼 파일**을 생성합니다.

### 핵심 원리

```
Isaac Sim Stage (metersPerUnit=1.0, upAxis=Z)
  └── USD 래퍼 (metersPerUnit=1.0, upAxis=Z)  ← 동일하므로 변환 없음!
        └── textured.obj (참조)
```

래퍼 USD의 `metersPerUnit`과 `upAxis`가 Isaac Sim stage와 동일하면, Metrics Assembler가 변환을 적용하지 않습니다.

### 코드

```python
from pxr import Usd, UsdGeom

# 1. 래퍼 USD 생성 (1회만)
wrapper_stage = Usd.Stage.CreateNew("textured_converted.usd")
UsdGeom.SetStageMetersPerUnit(wrapper_stage, 1.0)   # 미터 단위
UsdGeom.SetStageUpAxis(wrapper_stage, UsdGeom.Tokens.z)  # Z-up
root_prim = wrapper_stage.DefinePrim("/target_object", "Xform")
root_prim.GetReferences().AddReference("./textured.obj")  # 상대 경로
wrapper_stage.SetDefaultPrim(root_prim)
wrapper_stage.GetRootLayer().Save()

# 2. Isaac Sim에 로드
from isaacsim.core.utils.stage import add_reference_to_stage
add_reference_to_stage("textured_converted.usd", "/World/target_object")

# 3. 위치 설정 (기존 xformOp 사용)
prim = stage.GetPrimAtPath("/World/target_object")
prim.GetAttribute("xformOp:translate").Set(Gf.Vec3d(x, y, z))
```

> **주의:** `AddTranslateOp()`을 쓰면 이미 존재하는 op과 충돌합니다. 반드시 `GetAttribute("xformOp:translate").Set()`을 사용하세요.

## YCB 에셋 파일 구조

```
assets/ycb/
└── 002_master_chef_can/
    └── google_16k/
        ├── textured.obj           ← 메쉬 (필수)
        ├── textured.mtl           ← 재질 정의 (필수)
        ├── texture_map.png        ← 텍스처 (필수)
        └── textured_converted.usd ← USD 래퍼 (자동 생성)
```

### 필수 파일 (3개)
| 파일 | 용도 |
|------|------|
| `textured.obj` | 3D 메쉬 데이터 |
| `textured.mtl` | OBJ 재질 파일 (texture_map.png 참조) |
| `texture_map.png` | 텍스처 이미지 |

### 불필요한 파일 (삭제 가능)
| 파일 | 이유 |
|------|------|
| `kinbody.xml` | OpenRAVE 전용 |
| `nontextured.ply` | 텍스처 없는 메쉬 |
| `nontextured.stl` | 텍스처 없는 메쉬 |
| `textured.dae` | Collada 포맷 (미사용) |

## 다른 YCB 물체 추가하기

1. [YCB Dataset](https://www.ycbbenchmarks.com/)에서 다운로드
2. `assets/ycb/{물체번호_이름}/google_16k/`에 배치
3. 불필요 파일 삭제 (obj, mtl, png만 유지)
4. `receiver_isaac.py`에서 `OBJ_PATH` / `WRAPPER_USD_PATH` 경로 변경
5. 질량(`CreateMassAttr`) 업데이트
6. `textured_converted.usd`는 첫 실행 시 자동 생성됨

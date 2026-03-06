# Apple Vision Pro Teleoperation & Retargeting Pipeline

## 1. 개요 (Overview)
이 문서는 Apple Vision Pro(이하 VP)를 활용한 로봇 팔(Doosan) 및 아티큘레이티드 핸드(Allegro Hand)의 텔레오퍼레이션 파이프라인 구조를 설명합니다. 
본 파이프라인은 특히 `keti_retargeting` 레포지토리의 원리를 참고하여 최적화 및 간소화되었습니다.

## 2. Arm IK (Inverse Kinematics) 구조
VP에서 들어오는 오퍼레이터의 손목 자세(Wrist 6DoF Transform)를 바탕으로 로봇 팔의 끝단(End-Effector)을 제어하는 구조입니다.

### 2.1. 참고 및 적용 내용 (`keti_retargeting` 기반)
* **좌표계 변환 행렬 적용 (Coordinate Frame Transformation)**
  * VP의 좌표계(Camera/Head 기준)와 Isaac Sim 내의 로봇 월드 좌표계를 맞추기 위해 3x3 회전 변환 행렬 `OPERATOR2VP_RIGHT`를 곱해줍니다. 
  * `keti_retargeting`에서 사용된 방식을 채택하여 수식적 복잡함을 최소화하고 직관적인 방향 일치가 가능해졌습니다.
* **상대 좌표 매핑 (Relative Displacement Mapping)**
  * 오퍼레이터의 팔 길이와 로봇의 작업 반경이 다르기 때문에, 절대 좌표를 그대로 넘기면 로봇이 튀거나 범위를 이탈할 수 있습니다.
  * 시스템은 처음 손이 인식된 3D 위치를 영점(`calib_reference`)으로 잡고, 이후의 **이동 변화량(`vp_delta`)**만 계산해 로봇의 초기 설정된 작업 시작점(`INITIAL_EE_TARGET`)에 더합니다.

### 2.2 맞춤형 최적화 및 안전장치 (Safety Checks)
* **Gain Tuning**: `POS_GAIN = 1.6`배(이후 사용자 기호에 맞게 조정 가능)를 적용하여 오퍼레이터가 좁은 공간에서 팔을 덜 움직여도 로봇이 작업 데스크 전체에 도달할 수 있도록 편의성을 높였습니다.
* **공간 필터 (Workspace Soft Constraints)**:
  * 1) `Z_MIN_LIMIT` / `Z_CUSHION`: 바닥을 뚫고 들어가는 것을 막기 위해, 바닥에 가까워지면(10cm) 속도를 줄이는 Soft Cushion 알고리즘을 추가했으며 최하단 3cm 아래로는 절대 내려가지 못하게 Hard Clamp를 적용했습니다.
  * 2) `Low-pass Filter`: `alpha=0.3`의 이동 평균을 적용하여 사람 손목의 떨림이 로봇 모터에 무리를 주지 않도록 부드러운 궤적을 만듭니다.
* **Pitch Bias 보정**: 오퍼레이터가 테이블 위의 물체를 편하게 잡기 위해, 로봇 손이 기본적으로 바닥을 향해 **20도(-20 deg in Pitch) 기울어지도록** Quaternion 연산을 추가했습니다. (Target Quaternion \* Rx(20°)).
* **Lula Solver**: 계산된 최종 목표 위치/회전(Target Pose)을 Isaac Sim 내장 Lula Kinematics Solver에 전달하여 각 관절(arm_joint_1~6)의 안전한 각도를 역구동합니다.

---

## 3. Hand Retargeting 구조
VP의 3D 손가락 Joint(25개)를 Allegro Hand의 16개 구동 DOFs 사양으로 매핑하는 과정입니다.

### 3.1. 참고 및 적용 내용 (`keti_retargeting` 기반)
* **뼈대 벡터 기반 사잇각 연산 (Vector Geometry & Dot Product)**
  * 복잡한 Hand IK 최적화 솔버를 돌리는 대신, VP에서 주는 관절의 실제 3D 좌표를 빼서 뼈대 벡터(Bone Vectors)를 구한 후, 두 벡터 간의 사이각(`np.arccos(v1·v2)`)을 구하는 방식을 도입했습니다.
  * 이 방식은 단순명료하며 역기구학 특이점이나 연산 지연 없이 실시간(Real-time) 적용이 가능합니다.

### 3.2 맞춤형 튜닝 (Gains & Biases)
로봇 손과 사람 손의 생체 구조 차이(특히 알레그로 엄지의 구조)로 인해 사잇각만 1:1로 넘기면 물건을 꽉 쥘 수 없습니다.
* **손가락 굽힘(Flexion) 증폭**:
  * 사람 손가락의 주요 관절인 MCP, PIP 각도를 계산한 것에 `1.3배의 Gain`을 곱하여, 사람이 살짝만 쥐어도 로봇은 캔을 단단히 파지하도록 세팅했습니다.
* **엄지 손가락 궤적 혁신 (Thumb CMC Rotation)**:
  * 알레그로 엄지 첫 번째 축(CMC - Abduction 축)의 움직임을 인체와 동일하게 만들기 위해, "검지와 약지 메타카팔 뼈로 **손바닥 평면(Palm Plane)**의 법선 벡터를 구하고, 엄지 뼈 벡터가 이 평면 법선과 이루는 각도"를 계산했습니다.
  * 여기에 `1.5배` 증폭률을 적용하여 엄지가 자연스럽게 오퍼지션(Opposition, 다른 손가락과 맞닿음) 자세로 접힐 수 있게 되었습니다.

---

## 4. 제거 및 통폐합된 코드 (Cleanup)
- 이전 반복 실험에서 쓰이던 `quat_multiply_wxyz` 같은 저수준 직접 구현 수학 함수들은 제거되었고, `pxr.Gf.Quat`와 보정 로직 안으로 모두 통합되었습니다.
- 불필요하게 엄격해서 물체를 바닥에서 아예 집지 못하게 만들었던 강제적인 엘보우 높이 제한(FK Elbow check) 및 12cm 하드 Z-Clamp 로직을 걷어내고 Soft Cushion으로 교체하여 조작에 유연성을 확보했습니다.

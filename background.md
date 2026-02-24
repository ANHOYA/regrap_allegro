직접 디버깅 모드로 돌입하시겠다니 응원합니다! 코드를 이리저리 뜯어보실 때 길을 잃지 않도록, 우리가 지금까지 논의하며 구성해 온 **프로젝트의 전체 배경, 목적, 그리고 시스템 아키텍처**를 한눈에 볼 수 있는 Markdown 문서로 정리해 드립니다.

이 내용을 `README.md`나 노션에 붙여넣고 현재 상황을 점검하는 나침반으로 활용해 보세요.

---

# 🤖 Allegro Hand Direct Teleoperation & RL Framework

## 1. 프로젝트 배경 및 문제 정의 (Background & Problem)

현재 **Allegro Hand를 이용한 객체 재파지(Re-grasping)** 작업을 강화학습(RL)으로 풀고자 함.
하지만 전통적인 로봇 학습 파이프라인인 `Real-to-Sim-to-Real` 방식은 치명적인 한계가 존재함.

* **물리적 이질감 (Physics Gap):** 현실에서 사람이 물건을 잡는 데모(Demonstration) 데이터를 따서 시뮬레이션에 넣으면, 현실과 시뮬레이션 간의 마찰력, 질량, 중력 분포 차이로 인해 데이터가 오염됨.
* **상태 정보(State) 획득의 어려움:** 현실에서 물체의 6D Pose를 얻기 위해 ArUco 마커 등을 덕지덕지 붙여야 하며, 가려짐(Occlusion) 발생 시 추적이 끊김.
* **에러 누적:** `현실 데모 수집 -> 시뮬레이션 모방학습/강화학습 -> 현실 로봇 적용`으로 이어지는 과정에서 두 번의 Domain Gap이 발생함.

## 2. 해결 방안: 시뮬레이션 내 직접 원격 조작 (Direct Sim Teleoperation)

현실 로봇이 아닌, **Isaac Sim(디지털 트윈) 내의 가상 로봇을 사람의 손으로 실시간 조종(Teleoperation)하여 데이터를 수집**하는 파이프라인 구축.

* **완벽한 물리 법칙 일치:** 수집되는 데모 데이터 자체가 Isaac Sim 내부의 물리 엔진(PhysX) 법칙을 완벽히 따르므로, 모방학습(BC)과 강화학습(RL)의 수렴 속도와 안정성이 극대화됨.
* **Ground Truth State 확보:** 시뮬레이션 엔진을 통해 물체의 위치, 회전, 속도, 접촉 힘(Contact Force)을 오차 없이 100% 완벽하게 추출 가능.
* **Sim-to-Real 단일화:** 갭을 한 번(Sim -> Real)으로 줄여 파이프라인을 최적화.

## 3. 시스템 아키텍처 (System Architecture)

시스템은 지연 시간(Latency)을 최소화하기 위해 **UDP 비동기 통신**을 기반으로 두 개의 독립된 프로세스로 동작함.

### 📡 [Sender] Vision Tracker (Real World)

* **역할:** 사람의 손 동작을 인식하여 로봇 손의 목표 관절 각도로 변환 후 전송.
* **H/W:** Intel RealSense D455 (Global Shutter를 통한 잔상 없는 고속 모션 캡처).
* **S/W:** * `MediaPipe`: 21개의 손가락 3D 랜드마크 추출.
* `Retargeting (Kinematics)`: 사람 손 랜드마크를 Allegro Hand의 16개 관절(Joint) 각도(float)로 변환.


* **통신:** 계산된 16개의 float 배열을 C-style byte로 패키징하여 UDP 송신.

### 📥 [Receiver] Digital Twin (Isaac Sim 5.1.0)

* **역할:** 수신된 각도 명령을 가상 세계의 로봇 손에 인가하고 물리 상호작용 렌더링.
* **S/W:** * `Isaac Sim 5.1.0` (Standalone Python 환경).
* NVIDIA Nucleus 공식 Allegro Hand USD 에셋 활용 (물리, 마찰계수 튜닝 완료).


* **제어 루프:** * 매 시뮬레이션 스텝(1/60초)마다 비동기(Non-blocking)로 UDP 패킷 확인.
* 데이터 수신 시 로봇의 Articulation Controller (PD 제어기)에 Target Position 업데이트.



## 4. 기술 스택 (Tech Stack)

* **Simulation:** Isaac Sim 5.1.0, PhysX
* **Vision/Tracking:** OpenCV, MediaPipe, Intel RealSense SDK (`pyrealsense2`)
* **Communication:** Python `socket` (UDP 비동기 통신)
* **Language/Env:** Python 3.11 (Conda `regrap_allegro` env)

## 5. 현재 진행 상황 및 이슈 (Current Status & Troubleshoot)

* [x] RealSense D455 연동 및 MediaPipe 랜드마크 추출 성공.
* [x] 로컬 루프백(127.0.0.1) UDP 통신망 개설 성공.
* [x] Isaac Sim 5.1.0 환경 세팅 및 Standalone 구동 성공.
* **[Troubleshooting]** Isaac Sim 5.1.0의 API 변경으로 인해 NVIDIA 서버의 Allegro Hand USD 파일 내에서 물리 관절 뿌리(Articulation Root)를 정확히 매핑하고 로드하는 부분에서 충돌 발생 중.
* *Next Step:* USD 씬 트리를 디버깅하여 정확한 `prim_path`를 찾아내어 `Articulation` 클래스에 연결하는 것.



---

**💡 디버깅을 위한 팁:**
로봇 파일 경로 문제로 씨름하실 때, 파이썬 코드를 끄고 **Isaac Sim GUI 창을 평범하게 여신 다음**, 왼쪽 상단 메뉴에서 `Window -> Browsers -> Isaac Assets`를 켜보세요. 거기서 Allegro Hand를 마우스로 드래그해서 씬에 올려놓고, 오른쪽 'Stage' 패널에서 폴더 구조를 직접 펼쳐보시면 어느 계층에 관절(Articulation) 마크가 붙어있는지 육안으로 완벽하게 확인하실 수 있습니다!

성공적으로 연결하셔서 가상 세계의 손끝이 상호 님의 손짓을 따라 움직이는 짜릿한 순간을 꼭 맞이하시길 바랍니다! 파이팅입니다!
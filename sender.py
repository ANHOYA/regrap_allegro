import pyrealsense2 as rs
import cv2
import mediapipe as mp
import socket
import numpy as np
import struct

# 1. UDP 통신 세팅 (Isaac Sim이 실행될 PC의 IP와 Port)
UDP_IP = "127.0.0.1"  # 같은 PC에서 돌린다면 로컬호스트
UDP_PORT = 5005
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# 2. MediaPipe Hands 세팅
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
    max_num_hands=1
)

# 3. Intel RealSense D455 파이프라인 세팅
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60) # 60fps로 부드럽게
pipeline.start(config)

print(f"UDP 송신 시작: {UDP_IP}:{UDP_PORT}")

def angle_between_vectors(v1, v2):
    """두 벡터 사이의 각도 (라디안) 계산"""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)
    return np.arccos(cos_angle)

def get_finger_curl(landmarks, indices):
    """
    MediaPipe 랜드마크로부터 손가락 구부림 각도 3개를 계산.
    indices: [MCP, PIP, DIP, TIP] 인덱스 (예: [5,6,7,8] = 검지)
    반환: [MCP flexion, PIP flexion, DIP flexion]
    """
    pts = np.array([[landmarks[i].x, landmarks[i].y, landmarks[i].z] for i in indices])
    
    angles = []
    # 각 관절에서의 구부림 = 인접 두 뼈 벡터 사이 각도
    # 손목(0) → MCP, MCP → PIP, PIP → DIP, DIP → TIP
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    
    # MCP: 손목→MCP와 MCP→PIP 사이 각도
    v1 = pts[0] - wrist
    v2 = pts[1] - pts[0]
    angles.append(angle_between_vectors(v1, v2))
    
    # PIP: MCP→PIP와 PIP→DIP 사이 각도
    v1 = pts[1] - pts[0]
    v2 = pts[2] - pts[1]
    angles.append(angle_between_vectors(v1, v2))
    
    # DIP: PIP→DIP와 DIP→TIP 사이 각도
    v1 = pts[2] - pts[1]
    v2 = pts[3] - pts[2]
    angles.append(angle_between_vectors(v1, v2))
    
    return angles

def get_finger_abduction(landmarks, finger_mcp_idx, ref_mcp_idx):
    """
    손가락 벌림(abduction) 각도: 손바닥 평면에서 기준 손가락과의 벌어짐
    """
    wrist = np.array([landmarks[0].x, landmarks[0].y, landmarks[0].z])
    finger_mcp = np.array([landmarks[finger_mcp_idx].x, landmarks[finger_mcp_idx].y, landmarks[finger_mcp_idx].z])
    ref_mcp = np.array([landmarks[ref_mcp_idx].x, landmarks[ref_mcp_idx].y, landmarks[ref_mcp_idx].z])
    
    v_finger = finger_mcp - wrist
    v_ref = ref_mcp - wrist
    
    angle = angle_between_vectors(v_finger, v_ref)
    # 중앙 기준으로 부호 결정 (단순화: 양수만)
    return angle * 0.5  # 스케일링

def retarget_to_allegro(landmarks):
    """
    MediaPipe의 21개 3D 랜드마크를 Allegro Hand의 16개 joint angle로 변환.
    
    MediaPipe Landmarks:
      0: Wrist
      1-4: Thumb (CMC, MCP, IP, TIP)
      5-8: Index (MCP, PIP, DIP, TIP)
      9-12: Middle (MCP, PIP, DIP, TIP)
      13-16: Ring (MCP, PIP, DIP, TIP)
      17-20: Pinky (MCP, PIP, DIP, TIP)
    
    Allegro Hand Joints:
      0-3: Index  (abduction, MCP flexion, PIP flexion, DIP flexion)
      4-7: Middle (abduction, MCP flexion, PIP flexion, DIP flexion)
      8-11: Ring  (abduction, MCP flexion, PIP flexion, DIP flexion)
      12-15: Thumb (rotation, MCP flexion, PIP flexion, DIP flexion)
    """
    angles = np.zeros(16, dtype=np.float32)
    
    # === 검지 (joints 0-3) ===
    index_curl = get_finger_curl(landmarks, [5, 6, 7, 8])
    angles[0] = get_finger_abduction(landmarks, 5, 9)      # 벌림
    angles[1] = index_curl[0]                                 # MCP
    angles[2] = index_curl[1]                                 # PIP
    angles[3] = index_curl[2]                                 # DIP
    
    # === 중지 (joints 4-7) ===
    middle_curl = get_finger_curl(landmarks, [9, 10, 11, 12])
    angles[4] = get_finger_abduction(landmarks, 9, 5)       # 벌림
    angles[5] = middle_curl[0]
    angles[6] = middle_curl[1]
    angles[7] = middle_curl[2]
    
    # === 약지 (joints 8-11) — MediaPipe 약지 + 소지 반영 ===
    ring_curl = get_finger_curl(landmarks, [13, 14, 15, 16])
    pinky_curl = get_finger_curl(landmarks, [17, 18, 19, 20])
    angles[8] = get_finger_abduction(landmarks, 13, 9)      # 벌림
    angles[9] = (ring_curl[0] + pinky_curl[0]) / 2           # 약지+소지 평균
    angles[10] = (ring_curl[1] + pinky_curl[1]) / 2
    angles[11] = (ring_curl[2] + pinky_curl[2]) / 2
    
    # === 엄지 (joints 12-15) ===
    thumb_curl = get_finger_curl(landmarks, [1, 2, 3, 4])
    # 엄지 회전: 엄지 CMC와 검지 MCP 사이 벌림
    angles[12] = get_finger_abduction(landmarks, 1, 5)
    angles[13] = thumb_curl[0]
    angles[14] = thumb_curl[1]
    angles[15] = thumb_curl[2]
    
    # 각도를 Allegro joint limits에 맞게 클리핑
    # Abduction joints: roughly [-0.47, 0.47]
    # Flexion joints: roughly [-0.2, 1.7]
    for i in [0, 4, 8]:    # abduction
        angles[i] = np.clip(angles[i], -0.47, 0.47)
    for i in [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15]:  # flexion
        angles[i] = np.clip(angles[i], -0.2, 1.7)
    angles[12] = np.clip(angles[12], 0.263, 1.396)  # thumb rotation
    
    return angles

try:
    while True:
        # 프레임 받아오기
        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # OpenCV 이미지로 변환
        color_image = np.asanyarray(color_frame.get_data())
        image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)

        # MediaPipe로 손 추적
        results = hands.process(image_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 시각화 (화면에 관절 그리기)
                mp_drawing.draw_landmarks(color_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # 16개 Allegro 관절 각도 추출
                allegro_angles = retarget_to_allegro(hand_landmarks.landmark)
                
                # float32 배열(16개)을 바이트 데이터로 패키징하여 UDP 송신
                # struct.pack: 16개의 float(f)를 c 형식으로 변환 ('16f')
                data_bytes = struct.pack('16f', *allegro_angles)
                sock.sendto(data_bytes, (UDP_IP, UDP_PORT))
                
        # 화면 출력
        cv2.imshow('RealSense D455 - Teleoperation Sender', color_image)
        
        # 'q' 누르면 종료
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    pipeline.stop()
    cv2.destroyAllWindows()
    sock.close()
    print("통신 종료")
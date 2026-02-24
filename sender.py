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

def retarget_to_allegro(landmarks):
    """
    (핵심) MediaPipe의 21개 3D 랜드마크를 Allegro Hand의 16개 joint angle로 변환하는 함수.
    현재는 임시로 0.0 값이 들어가 있으며, 이 부분을 DexRetargeting 라이브러리로 교체하면 됩니다.
    """
    # TODO: DexRetargeting 로직 적용
    # 16개의 float 값 (Allegro Hand: Index, Middle, Ring, Thumb 순서 x 4 joint)
    dummy_angles = np.zeros(16, dtype=np.float32)
    
    # 임시 테스트용: 손목(0번)과 검지 끝(8번) 거리에 따라 검지 관절 각도 변화시키기
    # 손을 쥐면 값이 커지고, 펴면 작아지도록 간단한 테스트 로직
    y_dist = landmarks[8].y - landmarks[0].y
    dummy_angles[0:4] = y_dist * 2.0  # 검지 손가락 4개 관절에 임의의 값 인가
    
    return dummy_angles

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
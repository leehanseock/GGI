# yolo_camera.py
import cv2
from ultralytics import YOLO

# 모델 로드 (경로는 파일 위치에 맞게 수정)
model = YOLO("../yolo/yolov12n.pt")

# USB 카메라 열기 (기본 0번 카메라)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOv12 모델로 추론
    results = model(frame)

    # 결과 이미지 시각화
    annotated_frame = results[0].plot()

    # 화면 출력
    cv2.imshow("YOLOv12 Camera", annotated_frame)

    # 'q' 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()

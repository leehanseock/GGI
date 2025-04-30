from ultralytics import YOLO
import cv2

# 모델 로드 (너가 학습한 best.pt 경로)
model = YOLO("runs/detect/train11/weights/best.pt")

# USB 카메라 연결 (0번 카메라)
cap = cv2.VideoCapture(0)  # 내장 웹캠: 0, 외부 USB: 1 또는 그 이상

# 해상도 설정 (선택사항)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 루프 시작
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # 실시간 객체 감지 (conf 낮추면 약한 cigar도 감지 가능)
    results = model(frame, conf=0.25)

    # 결과 시각화
    annotated_frame = results[0].plot()

    # 화면에 출력
    cv2.imshow("YOLO Smoking Detection - Camera", annotated_frame)

    # 종료 키: q
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 카메라 및 창 종료
cap.release()
cv2.destroyAllWindows()

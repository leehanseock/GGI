from ultralytics import YOLO
import cv2

# 모델 로드
model = YOLO("runs/detect/train11/weights/best.pt")

# 테스트 이미지 경로
image_path = "/test_images/2.jpg"
image = cv2.imread(image_path)

# 추론
results = model.predict(source=image, conf=0.1, iou=0.4, show=True)

# ✅ 여기서 박스 결과를 출력!
for box in results[0].boxes:
    cls_id = int(box.cls)
    conf = float(box.conf)
    print(f"Class: {model.names[cls_id]}, Conf: {conf:.3f}")

# 결과 시각화
annotated = results[0].plot()

# 화면에 결과 표시
cv2.imshow("YOLO Smoking Detection - Image Test", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

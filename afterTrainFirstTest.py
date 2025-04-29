from ultralytics import YOLO
import cv2

# 학습된 모델 로드
model = YOLO("runs/detect/train7/weights/best.pt")

# 테스트할 이미지 경로
image_path = "D:/GGI/test_images/2.jpg"  # ← 원하는 이미지 경로로 수정하세요

# 이미지 읽기
image = cv2.imread(image_path)

# 추론 실행
results = model(image)

# 결과 시각화
annotated = results[0].plot()

# 결과 출력
cv2.imshow("YOLO Smoking Detection - Image Test", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()

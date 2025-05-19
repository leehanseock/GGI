from ultralytics import YOLO
import cv2

# YOLO 모델 로드
model = YOLO("runs/detect/train7/weights/best.pt")  # 너가 훈련한 모델 경로

# 테스트할 이미지 경로
image_path = "D:/GGI/test_images/3.jpg"  # 원하는 이미지 경로로 바꿔

# 이미지 불러오기
image = cv2.imread(image_path)

# YOLO 감지 실행
results = model(image, conf=0.25)

# 결과 시각화
annotated = results[0].plot()

# 결과 이미지 저장 (원하면 생략 가능)
cv2.imwrite("../output_result.jpg", annotated)

# 결과 출력
cv2.imshow("YOLO Image Detection", annotated)
cv2.waitKey(0)
cv2.destroyAllWindows()
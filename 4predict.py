from ultralytics import YOLO
from pathlib import Path

def main():
    # 훈련된 모델 불러오기 (best.pt)
    model = YOLO('runs/train/smoker_detector/weights/best.pt')

    # 테스트셋 이미지 디렉토리
    test_images_dir = Path('D:/GGI/test_images')

    # 추론 실행
    results = model.predict(
        source=str(test_images_dir),
        save=True,               # 결과 이미지 저장
        save_txt=True,           # 바운딩 박스 좌표 텍스트 저장
        conf=0.25,               # 신뢰도 임계값
        iou=0.45,                # NMS IOU 임계값
        imgsz=640,               # 입력 이미지 크기
        project='runs/predict', # 결과 저장 위치
        name='smoker_test',     # 결과 하위 폴더 이름
        exist_ok=True            # 이미 존재해도 덮어쓰기
    )

if __name__ == "__main__":
    main()

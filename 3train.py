from ultralytics import YOLO

def main():
    # 모델 불러오기
    model = YOLO('./yolo/yolov12n.pt')

    # 훈련 시작
    model.train(
        data='yolo/data.yaml',
        epochs=100,
        batch=16,
        name='smoker_detector',
        project='runs/train',
        pretrained=True,
        freeze=10,
        verbose=True,
        exist_ok=True  # 중요: 기존 폴더 덮어쓰기 허용
    )

if __name__ == "__main__":
    main()
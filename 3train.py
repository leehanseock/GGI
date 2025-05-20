from ultralytics import YOLO

def main():
    # 모델 불러오기
    model = YOLO('./yolo/yolov12n.pt')

    # 훈련 시작
    model.train(
        data='yolo/data.yaml',   # 절대경로면 더 안정적
        epochs=50,
        imgsz=640,
        batch=16,
        name='smoker_detector',
        project='runs/train',
        pretrained=True,
        freeze=10,
        verbose=True,
    )

if __name__ == "__main__":
    main()
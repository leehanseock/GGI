from ultralytics import YOLO

def main():
    # 전이학습 모델 불러오기 (yolov12n 기반으로 학습된 best0520.pt)
    model = YOLO('./yolo/best0520.pt')

    # 학습 시작
    model.train(
        data='yolo/data.yaml',
        cfg='yolo/augment.yaml',     # 💡 데이터 증강 설정
        epochs=100,
        batch=16,
        name='smoker_detector_second',
        project='runs/train',
        pretrained=True,
        freeze=0,
        verbose=True,
        exist_ok=True
    )

if __name__ == "__main__":
    main()


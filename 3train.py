from ultralytics import YOLO

def main():
    model = YOLO('./yolo/yolo11n.pt')  # 기존 모델에서 전이학습

    model.train(
        data='yolo/data.yaml',           # 수정된 클래스 구성 반영
        cfg='yolo/augment.yaml',         # 증강은 그대로 사용 가능
        epochs=100,
        batch=16,
        name='smoker_detector_cig_only',
        project='runs/train',
        pretrained=True,
        freeze=0,                        # 전체 학습
        verbose=True,
        exist_ok=True
    )

if __name__ == "__main__":
    main()
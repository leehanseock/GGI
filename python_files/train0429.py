from ultralytics import YOLO

if __name__ == "__main__":
    # 모델 불러오기
    model = YOLO('/yolo/yolov12n.pt')

    # 학습 시작
    model.train(
        data='D:/GGI/datasets/Smoking-CCTV-Detection.v1i.yolov12/data.yaml',
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        verbose=True  # 이거까지 추가했지
    )

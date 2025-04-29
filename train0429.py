from ultralytics import YOLO

# 모델 불러오기
model = YOLO('D:/GGI/yolo/yolov12n.pt')

# 학습 시작
model.train(
    data='D:/GGI/datasets/Smoking-CCTV-Detection.v1i.yolov12/data.yaml',
    epochs=100,
    imgsz=640,
    batch=32,
    device=0,
    verbose=True  # << 여기
)
from ultralytics import YOLO

# 기존 best.pt에서 이어서 fine-tuning
model = YOLO("runs/detect/train7/weights/best.pt")

# 학습 실행
model.train(
    data="D:/GGI/datasets/Smoking-CCTV.v1i.yolov11/data.yaml",
    epochs=50,
    patience=25,  # 25 epoch 동안 개선 없으면 중단
    cos_lr=True,
    batch=8,
    imgsz=640,
    device=0,
    workers=0,
    verbose=True,
    mosaic=True
)

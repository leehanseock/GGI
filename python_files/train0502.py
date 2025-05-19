from ultralytics import YOLO

def main():
    # 기존 best.pt에서 이어서 fine-tuning
    model = YOLO("runs/detect/train7/weights/best.pt")

    # 학습 실행
    model.train(
        data="D:/GGI/datasets/Smoking-CCTV.v1i.yolov11/data.yaml",
        epochs=50,
        patience=15,  # 15 epoch 동안 개선 없으면 종료 (25는 너무 길어질 수 있음)
        cos_lr=True,
        lr0=0.002,  # 기본 학습률 조정 (기본은 0.01 → 약간 줄여서 안정화)
        batch=16,
        imgsz=768,  # 작은 객체를 위한 고해상도 입력
        device=0,
        workers=2,
        cache=True,
        verbose=True,
        mosaic=True,
        degrees=5,  # 증강 설정들 추가 ↓
        scale=0.8,
        shear=2,
        perspective=0.0005
    )

if __name__ == "__main__":
    main()

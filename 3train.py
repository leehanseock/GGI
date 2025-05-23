from ultralytics import YOLO
import os

def main():
    # 모델 로딩
    model = YOLO('yolov8m.pt')  # 전이학습

    # 훈련
    model.train(
        data='D:/GGI/dataset_output/dataset.yaml',
        cfg='D:/GGI/yolo/augment.yaml',
        epochs=100,
        batch=32,
        name='smoker_detector',
        project='D:/GGI/runs/train',
        pretrained=True,
        freeze=0,
        verbose=True,
        exist_ok=True,
        patience=20,  # 🎯 mAP 향상 없으면 조기 중단 (default: 50 → 더 공격적으로 20)
    )

    # 훈련 로그 디렉토리
    exp_dir = model.trainer.save_dir  # ex: runs/train/smoker_detector

    # 훈련 결과 요약
    print(f"\n✅ 훈련 완료! 최종 결과 경로: {exp_dir}")
    print("📈 주요 파일:")
    print(f" - weights/best.pt (가장 성능 좋은 모델)")
    print(f" - results.png (정확도/손실 곡선)")
    print(f" - confusion_matrix.png (클래스 오차 시각화)")
    print(f" - TensorBoard: 실행하려면 👉  tensorboard --logdir {exp_dir}")

if __name__ == "__main__":
    main()

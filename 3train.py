from ultralytics import YOLO
import os

def main():
    # 이전 모델에서 이어서 학습!
    model = YOLO('D:/GGI/yolo/best052302.pt')  # 기존에 쓰던 best.pt 경로

    model.train(
        data='D:/GGI/dataset_output2/dataset.yaml',   # 새 데이터셋 yaml 경로
        cfg='D:/GGI/yolo/augment.yaml',               # 증강 설정(그대로 사용)
        epochs=100,
        batch=32,
        name='smoker_detector_resume_other_only',      # 새 실험명(구분용)
        project='D:/GGI/runs/train',
        pretrained=True,      # 이미 best.pt에서 이어서 하므로 True
        freeze=0,
        verbose=True,
        exist_ok=True,
        patience=20,          # mAP 향상 없으면 조기종료
    )

    exp_dir = model.trainer.save_dir

    print(f"\n✅ 훈련 완료! 최종 결과 경로: {exp_dir}")
    print("📈 주요 파일:")
    print(f" - weights/best.pt (가장 성능 좋은 모델)")
    print(f" - results.png (정확도/손실 곡선)")
    print(f" - confusion_matrix.png (클래스 오차 시각화)")
    print(f" - TensorBoard: 실행하려면 👉  tensorboard --logdir {exp_dir}")

if __name__ == "__main__":
    main()

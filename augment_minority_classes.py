import os
from pathlib import Path
import shutil

# 디렉토리 경로 설정
dataset_dir = Path("D:/GGI/dataset")  # 조정 가능
images_dir = dataset_dir / "frames_all"
labels_dir = dataset_dir / "labelled"

# 복제 대상 클래스 ID
target_classes = {'1', '2'}  # other, vape

# 증강용 복사 디렉토리
aug_images_dir = dataset_dir / "augmented/images"
aug_labels_dir = dataset_dir / "augmented/labels"
aug_images_dir.mkdir(parents=True, exist_ok=True)
aug_labels_dir.mkdir(parents=True, exist_ok=True)

count = 0
for label_file in labels_dir.glob("*.txt"):
    with open(label_file, "r") as f:
        lines = f.readlines()

    # 소수 클래스가 포함된 경우만 복제
    if any(line.strip().split()[0] in target_classes for line in lines):
        image_file = images_dir / (label_file.stem + ".jpg")
        if image_file.exists():
            shutil.copy(image_file, aug_images_dir / image_file.name)
            shutil.copy(label_file, aug_labels_dir / label_file.name)
            count += 1
        else:
            print(f"⚠️ 이미지 없음: {image_file.name}")

print(f"✅ 소수 클래스 이미지 및 라벨 {count}개 복제 완료 (→ {aug_images_dir})")
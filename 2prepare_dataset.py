import os
import shutil
import random
from pathlib import Path

# 기준 디렉토리 (GGI 내부에서 실행됨)
project_root = Path(__file__).parent
dataset_root = project_root / "dataset" / "dataset2"
images_dir = dataset_root / "frames" / "20250522_162442"  # ← 수정된 경로
labels_dir = dataset_root / "labelled"
base_output = project_root / "dataset2_output"

print("📁 이미지 디렉토리:", images_dir)
print("📁 라벨 디렉토리:", labels_dir)
print("📁 출력 디렉토리:", base_output)

splits = ['train', 'val', 'test']
split_ratio = [0.7, 0.2, 0.1]

# 출력 디렉토리 생성
for split in splits:
    (base_output / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (base_output / f"labels/{split}").mkdir(parents=True, exist_ok=True)

# 대상 이미지 파일 목록 중 라벨이 존재하는 것만 필터링
all_jpg_files = list(images_dir.glob("*.jpg"))
valid_image_files = []

for img_path in all_jpg_files:
    label_path = labels_dir / (img_path.stem + ".txt")
    if label_path.exists():
        valid_image_files.append(img_path)
    else:
        print(f"⚠️ 라벨 없음: {img_path.name}")

print(f"📦 라벨이 존재하는 이미지 수: {len(valid_image_files)}")

# 섞고 분할
random.shuffle(valid_image_files)
n = len(valid_image_files)
train_cutoff = int(n * split_ratio[0])
val_cutoff = train_cutoff + int(n * split_ratio[1])

split_files = {
    'train': valid_image_files[:train_cutoff],
    'val': valid_image_files[train_cutoff:val_cutoff],
    'test': valid_image_files[val_cutoff:]
}

# 파일 복사
for split, files in split_files.items():
    for img_path in files:
        label_path = labels_dir / (img_path.stem + ".txt")
        shutil.copy(img_path, base_output / f"images/{split}" / img_path.name)
        shutil.copy(label_path, base_output / f"labels/{split}" / label_path.name)

print("✅ 데이터셋 분할 완료 (라벨 없는 이미지 제외됨)")
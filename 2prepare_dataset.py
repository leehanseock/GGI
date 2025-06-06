import os
import shutil
import random
from pathlib import Path

# 경로 설정 (환경에 맞게 수정!)
project_root = Path(__file__).parent
dataset_root = project_root / "dataset"
images_dir = dataset_root / "frames_all0"     # ✅ 통합 이미지 폴더
labels_dir = dataset_root / "labelled0"       # ✅ 통합 라벨 폴더
base_output = project_root / "dataset_output3"  # ✅ 출력 폴더 (새로 만드세요!)

print("📁 이미지 디렉토리:", images_dir)
print("📁 라벨 디렉토리:", labels_dir)
print("📁 출력 디렉토리:", base_output)

splits = ['train', 'val', 'test']
split_ratio = [0.7, 0.2, 0.1]  # 필요시 수정

# 출력 폴더 생성
for split in splits:
    (base_output / f"images/{split}").mkdir(parents=True, exist_ok=True)
    (base_output / f"labels/{split}").mkdir(parents=True, exist_ok=True)

all_jpg_files = list(images_dir.glob("*.jpg"))
valid_image_files = []

for img_path in all_jpg_files:
    label_path = labels_dir / (img_path.stem + ".txt")
    if label_path.exists():
        with open(label_path, "r") as f:
            lines = f.readlines()
        # 라벨이 1줄 이상 있으면 추가 (클래스 제한 없이 통합)
        if len(lines) > 0:
            valid_image_files.append(img_path)
        else:
            print(f"⚠️ 빈 라벨: {img_path.name}")
    else:
        print(f"⚠️ 라벨 없음: {img_path.name}")

print(f"📦 유효한 라벨이 있는 이미지 수: {len(valid_image_files)}")

# 셔플 및 분할
random.shuffle(valid_image_files)
n = len(valid_image_files)
train_cutoff = int(n * split_ratio[0])
val_cutoff = train_cutoff + int(n * split_ratio[1])

split_files = {
    'train': valid_image_files[:train_cutoff],
    'val': valid_image_files[train_cutoff:val_cutoff],
    'test': valid_image_files[val_cutoff:]
}

for split, files in split_files.items():
    for img_path in files:
        label_path = labels_dir / (img_path.stem + ".txt")
        shutil.copy(img_path, base_output / f"images/{split}" / img_path.name)
        shutil.copy(label_path, base_output / f"labels/{split}" / label_path.name)

print("✅ 데이터셋 분할 완료 (모든 클래스 포함)")

# dataset.yaml 생성 (클래스명, 개수 기존과 일치)
class_names = ["cigarette", "other", "vape"]  # 반드시 기존 모델과 일치!

dataset_yaml_path = base_output / "dataset.yaml"
with open(dataset_yaml_path, "w") as f:
    f.write("train: " + str(base_output / "images/train").replace("\\", "/") + "\n")
    f.write("val: " + str(base_output / "images/val").replace("\\", "/") + "\n")
    f.write("test: " + str(base_output / "images/test").replace("\\", "/") + "\n\n")
    f.write(f"nc: {len(class_names)}\n")
    f.write("names: " + str(class_names) + "\n")

print(f"✅ dataset.yaml 파일 생성 완료: {dataset_yaml_path}")

import os
from pathlib import Path

# 대상 디렉토리
target_dir = Path("D:/GGI/dataset/frames/WIN_20250520_17_37_41_Pro")
image_exts = [".jpg", ".jpeg", ".png"]
prefix = "vape_"  # 새로운 파일 이름 접두어
start_index = 0   # 시작 인덱스

# 대상 디렉토리 내 이미지 파일만 추출
image_files = sorted([f for f in target_dir.iterdir() if f.suffix.lower() in image_exts])

for idx, img_file in enumerate(image_files, start=start_index):
    new_name = f"{prefix}{idx:04d}{img_file.suffix.lower()}"
    new_path = target_dir / new_name
    img_file.rename(new_path)

print(f"✅ 이미지 {len(image_files)}개 이름 변경 완료 (라벨 파일은 그대로 유지됨)")

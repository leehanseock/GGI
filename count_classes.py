from pathlib import Path
from collections import Counter

# 클래스 이름 매핑
class_map = {
    "cigarette": "0",
    "other": "1"
}

label_dir = Path("D:/GGI/dataset/labelled")
label_files = list(label_dir.glob("*.txt"))

converted = 0
skipped_lines = 0

for file in label_files:
    updated_lines = []
    with open(file, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue

            if parts[0] in class_map:
                parts[0] = class_map[parts[0]]  # 문자 → 숫자
            elif not parts[0].isdigit():
                print(f"⚠️ 잘못된 클래스 라벨 '{parts[0]}' in {file.name}")
                skipped_lines += 1
                continue

            updated_lines.append(" ".join(parts))

    if updated_lines:
        with open(file, 'w') as f:
            f.write("\n".join(updated_lines) + "\n")
        converted += 1

print(f"✅ 변환 완료: {converted}개 파일 수정됨")
if skipped_lines:
    print(f"⚠️ 처리되지 않은 라벨 {skipped_lines}개 있음")

# ------- 클래스 수 집계 --------
def count_classes_in_labels(label_dir):
    label_dir = Path(label_dir)
    label_files = list(label_dir.rglob("*.txt"))
    class_counts = Counter()

    for label_file in label_files:
        with open(label_file, 'r') as f:
            for line in f:
                if line.strip() == "":
                    continue
                class_id = line.strip().split()[0]
                if not class_id.isdigit():
                    print(f"❌ 숫자가 아닌 클래스 ID 발견: '{class_id}' in {label_file.name}")
                    continue
                class_counts[int(class_id)] += 1

    return class_counts

if __name__ == "__main__":
    counts = count_classes_in_labels(label_dir)

    class_names = {0: "cigarette", 1: "other"}

    print("📊 클래스별 객체 수:")
    for cls_id, count in sorted(counts.items()):
        name = class_names.get(cls_id, f"class_{cls_id}")
        print(f"  {name} ({cls_id}): {count}개")

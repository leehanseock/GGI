import os
from collections import Counter

# 라벨 파일 경로
label_dir = "D:/GGI/dataset/labelled"

# 클래스 ID → 이름 매핑
class_map = {
    "0": "cigarette",
    "1": "other",
    "2": "vape"
}

class_counts = Counter()
total_labels = 0

for filename in os.listdir(label_dir):
    if filename.endswith(".txt"):
        file_path = os.path.join(label_dir, filename)
        with open(file_path, "r") as f:
            lines = f.readlines()
            total_labels += len(lines)
            for line in lines:
                if line.strip():
                    raw_id = line.strip().split()[0]
                    class_name = class_map.get(raw_id, raw_id)  # 숫자면 이름으로 변환, 없으면 그대로 사용
                    class_counts[class_name] += 1

# 출력
print(f"총 라벨 수: {total_labels}")
print("클래스별 개수:")
for class_name, count in sorted(class_counts.items()):
    print(f"  클래스 '{class_name}': {count}개")

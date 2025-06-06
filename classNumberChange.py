from pathlib import Path

labels_dir = Path("D:/GGI/dataset/labelled2")   # 라벨 txt 폴더
output_dir = Path("D:/GGI/dataset/labelled2_fixed")  # 변환된 라벨 저장 폴더

output_dir.mkdir(parents=True, exist_ok=True)

for txt_file in labels_dir.glob("*.txt"):
    with open(txt_file, "r") as f:
        lines = f.readlines()
    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if parts:  # 빈 줄 방지
            # 맨 앞이 0이면 1로 바꿔줌
            if parts[0] == "0":
                parts[0] = "1"
            new_lines.append(" ".join(parts))
    # 새 폴더에 저장
    with open(output_dir / txt_file.name, "w") as f:
        f.write("\n".join(new_lines) + "\n")

print("✅ 클래스 번호 0 → 1 변환 완료! (새 폴더에 저장)")
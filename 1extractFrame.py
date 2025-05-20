import cv2
import os
import sys

# 비디오 파일 경로 설정
video_path = os.path.join("D:/GGI/dataset/files", "WIN_20250520_17_37_41_Pro.mp4")

# 비디오 파일명만 추출 (확장자 제거)
video_name = os.path.splitext(os.path.basename(video_path))[0]

# 출력 디렉토리 설정: frames/영상이름/
output_base_dir = os.path.join("D:/GGI/dataset", "frames")
output_dir = os.path.join(output_base_dir, video_name)

# 출력 디렉토리 생성
os.makedirs(output_dir, exist_ok=True)

# 비디오 캡처 객체 생성
cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    print("비디오 파일을 열 수 없습니다.")
    exit()

# 전체 프레임 수 가져오기
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
if total_frames == 0:
    print("비디오에 프레임이 없습니다.")
    cap.release()
    exit()

frame_count = 0
print(f"총 {total_frames}개의 프레임을 저장합니다...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_filename = f"frame_{frame_count:04d}.jpg"
    frame_path = os.path.join(output_dir, frame_filename)

    success = cv2.imwrite(frame_path, frame)
    if not success:
        print(f"{frame_path} 저장 실패")

    frame_count += 1

    # 진행률 출력
    progress = (frame_count / total_frames) * 100
    sys.stdout.write(f"\r진행률: {progress:.2f}%")
    sys.stdout.flush()

cap.release()
print(f"\n총 {frame_count}개의 프레임이 '{output_dir}'에 저장되었습니다.")

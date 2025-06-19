import cv2
import os
import sys

# 비디오 파일들이 위치한 디렉토리
video_dir = "D:/GGI/dataset/files"
video_files = ["1.mp4","2.mp4", "3.mp4", "4.mp4", "5.mp4", "6.mp4"]

# 모든 프레임을 저장할 하나의 출력 디렉토리
output_dir = "D:/GGI/dataset/frames_all"
os.makedirs(output_dir, exist_ok=True)

global_frame_index = 0  # 전체 프레임 번호

save_interval = 30  # 0.5초 간격 (60fps 기준)

for video_file in video_files:
    video_path = os.path.join(video_dir, video_file)

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"비디오 파일을 열 수 없습니다: {video_file}")
        continue

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print(f"비디오에 프레임이 없습니다: {video_file}")
        cap.release()
        continue

    frame_count = 0
    saved_count = 0

    print(f"\n'{video_file}': 총 {total_frames}프레임 중, {save_interval}프레임마다 저장합니다...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % save_interval == 0:
            frame_filename = f"frame_{global_frame_index:06d}.jpg"
            frame_path = os.path.join(output_dir, frame_filename)
            success = cv2.imwrite(frame_path, frame)
            if success:
                saved_count += 1
                global_frame_index += 1
            else:
                print(f"{frame_path} 저장 실패")

        frame_count += 1

        progress = (frame_count / total_frames) * 100
        sys.stdout.write(f"\r[{video_file}] 진행률: {progress:.2f}%")
        sys.stdout.flush()

    cap.release()
    print(f"\n'{video_file}'에서 총 {saved_count}개의 프레임이 저장되었습니다.")

print(f"\n총 {global_frame_index}개의 프레임이 '{output_dir}'에 저장되었습니다.")
from ultralytics import YOLO

#웹캠 테스트
def main():
    # 훈련된 모델 불러오기
    model = YOLO('D:/GGI/yolo/best0606.pt')

    # 웹캠에서 실시간 감지 (기본 웹캠은 source=0)
    results = model.predict(
        source=0,                # ✅ 웹캠 번호 (0번 웹캠 기본)
        conf=0.1,                # 신뢰도 임계값
        iou=0.45,                # NMS IOU 임계값
        imgsz=640,               # 입력 이미지 크기
        show=True,               # ✅ 창에 실시간 출력
        save=True,               # 결과 저장
        save_txt=True,           # 바운딩 박스 좌표 저장
        project='runs/predict',
        name='smoker_webcam',
        exist_ok=True
    )

if __name__ == "__main__":
    main()

# # 이미지 테스트
# from pathlib import Path
# import shutil

# # 경로 설정
# model_path = "D:/GGI/yolo/best0606.pt"
# input_dir = Path("D:/GGI/test_images")
# output_dir = Path("D:/GGI/test_images/test_results")
# output_dir.mkdir(parents=True, exist_ok=True)
#
# # 모델 로딩
# model = YOLO(model_path)
#
# # 이미지들 추론
# results = model.predict(
#     source=str(input_dir),
#     save=True,                   # 이미지 저장
#     save_txt=False,              # 라벨 텍스트 저장 원할 경우 True
#     project=str(output_dir),     # 결과 저장 경로
#     name="",                     # 하위 폴더 없이 바로 저장
#     exist_ok=True,
#     conf=0.5                     # confidence threshold (필요 시 조절)
# )
#
# print(f"✅ 추론 완료: 결과가 {output_dir}에 저장되었습니다.")

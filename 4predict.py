from ultralytics import YOLO

def main():
    # 훈련된 모델 불러오기
    model = YOLO('D:/GGI/yolo/best0523.pt')

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
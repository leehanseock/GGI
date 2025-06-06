# 모니터 화면에서 객체 인식 테스트
import mss
import numpy as np
import cv2
from ultralytics import YOLO

model = YOLO('D:/GGI/yolo/best0606.pt')

monitor = {'top': 100, 'left': 100, 'width': 800, 'height': 600}  # 원하는 영역 지정

with mss.mss() as sct:
    while True:
        img = np.array(sct.grab(monitor))
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        results = model.predict(img, conf=0.4)
        out_img = results[0].plot()
        cv2.imshow("Screen Detection", out_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
cv2.destroyAllWindows()

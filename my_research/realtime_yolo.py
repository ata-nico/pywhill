
import cv2
from ultralytics import YOLO

# YOLOv8のモデルをロード
model = YOLO(r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans8\weights\best.pt") 

# Webカメラの起動
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # YOLOで物体検出を行う
    results = model(frame)

    # 結果をフレームに描画して表示
    annotated_frame = results[0].plot()

    # ウィンドウに描画した結果を表示
    cv2.imshow("YOLO Detection", annotated_frame)

    # 'q'を押すと終了
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


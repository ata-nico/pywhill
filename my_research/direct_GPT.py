import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
from datetime import datetime

# --- 設定 ---
IMAGE_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\panorama\panorama_img\yolo_002_18.jpg"      # 入力画像パス
MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans17\weights\best.pt"       # YOLOモデルパス
ROI_ANGLE_DEG    = 60                         # 中心部±ROI_ANGLE [deg]
FOV_DEG      = 360                        # パノラマの水平視野角
CONF_THRESH  = 0.2                        # YOLO検出信頼度閾値
ANGLE_THRESH_DEG = 5                        # 平均角度による直進/旋回判定閾値（°）
OUTPUT_DIR   = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute"    # 保存先ディレクトリ

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
# 保存ファイル用タイムスタンプ
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 1. 画像読み込み & モデルロード ---
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]
model = YOLO(MODEL_PATH)

# --- 2. YOLO推論 ---
results = model.predict(img, conf=CONF_THRESH, imgsz=(w, h))
boxes = results[0].boxes.xyxy.cpu().numpy()

# --- 3. ROIフィルタリング & 水平角度算出 ---
angles_deg = []
filtered = []
for x1, y1, x2, y2 in boxes:
    cx = (x1 + x2) / 2
    # ピクセル偏差 → 水平角度(°) の線形マッピング
    theta_deg = (cx - w/2) / w * FOV_DEG
    if abs(theta_deg) <= ROI_ANGLE_DEG:
        angles_deg.append(theta_deg)
        filtered.append((int(x1), int(y1), int(x2), int(y2)))

# --- 4. 進行方向判定 ---
if angles_deg:
    mean_ang = sum(angles_deg) / len(angles_deg)
    direction = 'straight' if abs(mean_ang) < ANGLE_THRESH_DEG else 'turn'
else:
    mean_ang = 0
    direction = 'no detection'

# --- 5. 描画 ---
# バウンディングボックスと判定結果
for x1, y1, x2, y2 in filtered:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
text = f"Dir: {direction} ({mean_ang:.1f}°)"
cv2.putText(img, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

# ROI領域表示（画像中央±ROI）
roi_x1 = int((w/2) - (ROI_ANGLE_DEG/ FOV_DEG) * w)
roi_x2 = int((w/2) + (ROI_ANGLE_DEG/ FOV_DEG) * w)
roi = img[:, roi_x1:roi_x2]
cv2.putText(roi, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

# --- 6. 保存 ---
pano_out = os.path.join(OUTPUT_DIR, f"panorama_{timestamp}.jpg")
roi_out  = os.path.join(OUTPUT_DIR, f"roi_{timestamp}.jpg")
cv2.imwrite(pano_out, img)
cv2.imwrite(roi_out, roi)

# --- 7. 表示 ---
cv2.imshow("Annotated Panorama", img)
cv2.imshow("ROI", roi)
print(f"Saved: {pano_out}, {roi_out}")
cv2.waitKey(0)
cv2.destroyAllWindows()

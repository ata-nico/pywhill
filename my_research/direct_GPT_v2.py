import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
from datetime import datetime

# --- 設定 ---
IMAGE_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\panorama\panorama_img\yolo_002_20.jpg"      # 入力画像パス
MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans17\weights\best.pt"       # YOLOモデルパス
ROI_ANGLE_DEG    = 60                         # 中心部±ROI_ANGLE [deg]
FOV_DEG      = 360                        # パノラマの水平視野角
CONF_THRESH  = 0.2                        # YOLO検出信頼度閾値
RSS_THRESH       = 10000                   # RSS(残差平方和)による判定閾値
OUTPUT_DIR   = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute"    # 保存先ディレクトリ

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
# タイムスタンプ取得 (ファイル名用)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 1. 画像読み込み & モデルロード ---
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]
model = YOLO(MODEL_PATH)

# --- 2. YOLO推論 ---
results = model.predict(img, conf=CONF_THRESH, imgsz=(w, h))
boxes = results[0].boxes.xyxy.cpu().numpy()

# --- 3. ROIフィルタリング & 重心算出 ---
centers = []
for x1, y1, x2, y2 in boxes:
    cx = (x1 + x2) / 2
    # 水平角度計算 (deg)
    theta_deg = (cx - w/2) / w * FOV_DEG
    if abs(theta_deg) <= ROI_ANGLE_DEG:
        cy = (y1 + y2) / 2
        centers.append((cx, cy, x1, y1, x2, y2))

# --- 4. 最小二乗直線フィット & 垂直距離RSS計算 ---
direction = 'no detection'
rss = 0.0

if len(centers) >= 2:
    data = np.array([[c[0], c[1]] for c in centers])  # shape (N,2)
    X = np.vstack([data[:,0], np.ones(len(data))]).T   # shape (N,2)
    Y = data[:,1]
    # 最小二乗で直線 y = m*x + b
    m, b = np.linalg.lstsq(X, Y, rcond=None)[0]

    # 直線への垂直距離を計算
    # d_i = |m x_i - y_i + b| / sqrt(m^2 + 1)
    numer = np.abs(m*data[:,0] - data[:,1] + b)
    denom = math.sqrt(m*m + 1)
    distances = numer / denom

    # 二乗和を RSS として扱う
    rss = float(np.sum(distances**2))
    direction = 'straight' if rss < RSS_THRESH else 'turn'


# --- 5. 描画 ---
# バウンディングボックスと方向テキスト
for cx, cy, x1, y1, x2, y2 in centers:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 2)
text = f"Dir: {direction} (RSS={rss:.0f})"
cv2.putText(img, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

# ROI範囲を画像上表示
roi_x1 = int((w/2) - (ROI_ANGLE_DEG/ FOV_DEG) * w)
roi_x2 = int((w/2) + (ROI_ANGLE_DEG/ FOV_DEG) * w)
cv2.rectangle(img, (roi_x1, 0), (roi_x2, h), (255,0,0), 2)
roi = img[:, roi_x1:roi_x2]
cv2.putText(roi, text, (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,0,0), 2)

# --- 6. 保存 ---
pano_out = os.path.join(OUTPUT_DIR, f"panorama_{timestamp}.jpg")
roi_out  = os.path.join(OUTPUT_DIR, f"roi_{timestamp}.jpg")
cv2.imwrite(pano_out, img)
cv2.imwrite(roi_out, roi)

# --- 7. 結果表示 ---
print(f"Saved: {pano_out}, {roi_out}")
cv2.imshow("Annotated Panorama", img)
cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()
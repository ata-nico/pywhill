import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
from datetime import datetime
from sklearn.linear_model import RANSACRegressor, LinearRegression

# --- 設定 ---
IMAGE_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\panorama\panorama_img\yolo_003_30.jpg"      # 入力画像パス
MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans17\weights\best.pt"       # YOLOモデルパス
ROI_ANGLE_DEG    = 70                      # 中心部±ROI角度 [deg]
FOV_DEG          = 360                      # パノラマの水平視野角
CONF_THRESH      = 0.2                      # YOLO検出信頼度閾値
RANSAC_THRESH_PX = 5.0                      # RANSAC の residual_threshold (px)
INLIER_RATIO_TH  = 0.7                      # インライア比率閾値
OUTPUT_DIR   = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute"    # 保存先ディレクトリ

# 出力ディレクトリ準備
os.makedirs(OUTPUT_DIR, exist_ok=True)
# タイムスタンプ取得
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 1. 画像読み込み & モデルロード ---
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]
model = YOLO(MODEL_PATH)

# --- 2. YOLO 推論 ---
results = model.predict(img, conf=CONF_THRESH, imgsz=(w, h))
boxes = results[0].boxes.xyxy.cpu().numpy()  # x1,y1,x2,y2

# --- 3. ROI と重心抽出 ---
centers = []
for x1, y1, x2, y2 in boxes:
    cx = (x1 + x2) / 2
    # ピクセル偏差→水平角度(°)
    theta_deg = (cx - w/2) / w * FOV_DEG
    if abs(theta_deg) <= ROI_ANGLE_DEG:
        cy = (y1 + y2) / 2
        centers.append((cx, cy, int(x1), int(y1), int(x2), int(y2)))

# ROI 領域の切り出し
roi_x1 = int((w/2) - (ROI_ANGLE_DEG / FOV_DEG) * w)
roi_x2 = int((w/2) + (ROI_ANGLE_DEG / FOV_DEG) * w)
roi = img[:, roi_x1:roi_x2]

# --- 4. RANSAC 直線フィット & インライア比率計算 ---
direction = 'no detection'
inlier_ratio = 0.0
if len(centers) >= 2:
    # 点群配列
    data = np.array([[c[0], c[1]] for c in centers])  # shape (N,2)
    X = data[:,0].reshape(-1, 1)  # x 座標
    Y = data[:,1]                # y 座標
    # RANSACRegressor
    ransac = RANSACRegressor(LinearRegression(),
                            residual_threshold=RANSAC_THRESH_PX,
                            random_state=0)
    ransac.fit(X, Y)
    inlier_mask = ransac.inlier_mask_
    inlier_ratio = inlier_mask.sum() / len(data)
    direction = 'straight' if inlier_ratio >= INLIER_RATIO_TH else 'turn'

# --- 5. 描画 ---
# 全パノラマに検出ボックス
for cx, cy, x1, y1, x2, y2 in centers:
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
# ROI枠を描画
cv2.rectangle(img, (roi_x1, 0), (roi_x2, h), (255, 0, 0), 2)
# 方向テキスト描画
text = f"Dir: {direction} (Inlier: {inlier_ratio:.2f})"
cv2.putText(img, text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
cv2.putText(roi, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)

# --- 6. 出力保存 ---
pano_out = os.path.join(OUTPUT_DIR, f"panorama_{timestamp}.jpg")
roi_out  = os.path.join(OUTPUT_DIR, f"roi_{timestamp}.jpg")
cv2.imwrite(pano_out, img)
cv2.imwrite(roi_out, roi)

# --- 7. 表示 & 終了 ---
print(f"Saved: {pano_out}, {roi_out}")
cv2.imshow("Annotated Panorama", img)
cv2.imshow("ROI", roi)
cv2.waitKey(0)
cv2.destroyAllWindows()

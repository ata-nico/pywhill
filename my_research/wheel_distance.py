import cv2
import numpy as np
from ultralytics import YOLO
import math
import os
from datetime import datetime

# --- 設定 ---
IMAGE_PATH     = r"C:\Users\ata3357\Desktop\zemi_win\panorama\panorama_img\yolo_001_8.jpg"
MODEL_PATH     = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans\weights\best.pt"
ROI_ANGLE_DEG  = 60                          # 中心部±ROI_ANGLE [deg]
FOV_DEG        = 360                         # パノラマの水平視野角 [deg]
CONF_THRESH    = 0.2                         # YOLO検出信頼度閾値
RSS_THRESH     = 10000                      # RSS(残差平方和)閾値
REAL_WIDTH     = 0.2                         # 点字ブロック実物幅 [m]
SIDE_THRESH_DEG= 10                          # 真横判定許容角度幅 [deg]
CAM_H          = 1.55                       # カメラ地上高 [m]
OUTPUT_DIR     = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute"

# 出力ディレクトリ作成
os.makedirs(OUTPUT_DIR, exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# --- 1. 読み込み & モデルロード ---
img = cv2.imread(IMAGE_PATH)
h, w = img.shape[:2]
model = YOLO(MODEL_PATH)

# --- 2. YOLO推論 ---
results = model.predict(img, conf=CONF_THRESH, imgsz=(w, h))
boxes = results[0].boxes.xyxy.cpu().numpy()

# --- 3. ROIと真横ブロック検出 & 地面距離推定 ---
centers = []          # ROI内センター
side_info = []        # 真横ブロック情報
for x1, y1, x2, y2 in boxes:
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2  # 重心の y 座標
    # 水平角度 (deg)
    theta_h_deg = ((cx - w/2) / w) * FOV_DEG
    # ROI内
    if abs(theta_h_deg) <= ROI_ANGLE_DEG:
        centers.append((x1, y1, x2, y2))
    # 真横 (±90° ± SIDE_THRESH)
    if abs(abs(theta_h_deg) - 90.0) <= SIDE_THRESH_DEG:
        # 地面距離推定: 最大 y 値を使用
        y_max = max(y1, y2, cy)
        phi_rad = ((y_max / h) - 0.5) * math.pi  # -π/2 to +π/2
        if abs(phi_rad) < 1e-3:
            D_ground = float('inf')
        else:
            D_ground = CAM_H / math.tan(abs(phi_rad))
        side_info.append((x1, y1, x2, y2, D_ground))

# --- 4. RSS判定 ---
direction = 'no detection'
rss = 0.0
if len(centers) >= 2:
    data = np.array([[ (x1+x2)/2, (y1+y2)/2 ] for x1,y1,x2,y2 in centers])
    X = np.vstack([data[:,0], np.ones(len(data))]).T
    Y = data[:,1]
    m, b = np.linalg.lstsq(X, Y, rcond=None)[0]
    dist_line = np.abs(m*data[:,0] - data[:,1] + b) / math.sqrt(m*m + 1)
    rss = float(np.sum(dist_line**2))
    direction = 'straight' if rss < RSS_THRESH else 'turn'

# --- 5. 描画: 地面距離のみ ---
# 真横ブロック: 緑枠 & 地面距離表示
for x1, y1, x2, y2, Dg in side_info:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,0), 3)
    cv2.putText(img, f"G:{Dg:.2f}m", (int(x1), int(y1)-5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
# ROI枠: 青
roi_x1 = int((w/2) - (ROI_ANGLE_DEG / FOV_DEG) * w)
roi_x2 = int((w/2) + (ROI_ANGLE_DEG / FOV_DEG) * w)
cv2.rectangle(img, (roi_x1, 0), (roi_x2, h), (255,0,0), 2)
# ROI内検出: 白枠
for x1, y1, x2, y2 in centers:
    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (255,255,255), 1)
# テキスト
text = f"Dir: {direction}  RSS: {int(rss)}"
cv2.putText(img, text, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255,255,255), 2)

# --- 6. 保存 & 表示 ---
pano_out = os.path.join(OUTPUT_DIR, f"panorama_{timestamp}.jpg")
cv2.imwrite(pano_out, img)
print(f"Saved: {pano_out}")
cv2.imshow("Annotated Panorama", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

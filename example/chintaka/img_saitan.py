import cv2
import numpy as np
import math
from ultralytics import YOLO

# --- 設定 ---
MODEL_PATH  = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt"
IMAGE_PATH  = r"C:\Users\ata3357\Desktop\zemi_win\panorama\panorama_img\yolo_003_13.jpg"  # ← ここに画像ファイルパスを指定
CONF_THRESH = 0.2

# --- デバッグ描画関数 ---
def draw_debug(frame, bx, by, theta_lat):
    h, w = frame.shape[:2]
    center = (w//2, h//2)
    length = max(w, h)
    # 真横ラインの終点
    dx = math.cos(theta_lat) * length
    dy = math.sin(theta_lat) * length
    hori_end = (int(center[0] + dx), int(center[1] + dy))
    blk_end = (int(bx), int(by))
    cv2.circle(frame, center, 5, (0,255,0), -1)          # カメラ中心
    cv2.circle(frame, blk_end, 5, (0,0,255), -1)         # ブロック重心
    cv2.line(frame, center, hori_end, (255,0,0), 2)      # 真横ライン (青)
    cv2.line(frame, center, blk_end, (0,255,255), 2)     # 最短距離ライン (黄)
    return frame

# --- メイン処理 ---
model = YOLO(MODEL_PATH)
frame = cv2.imread(IMAGE_PATH)
if frame is None:
    raise FileNotFoundError(f"画像が見つかりません: {IMAGE_PATH}")

h, w = frame.shape[:2]
# YOLO検出
det = model(frame, conf=CONF_THRESH)[0]
centers = []
for box in det.boxes:
    x1, y1, x2, y2 = box.xyxy.cpu().numpy().flatten()
    cx, cy = (x1+x2)/2, (y1+y2)/2
    centers.append((cx, cy))

if centers:
    # “最短距離ブロック” を y 最大で選択
    bx, by = max(centers, key=lambda c: c[1])
    # 真横方向（x>=中心→右、左→左）
    theta_lat = math.pi/2 if bx >= w/2 else -math.pi/2
    frame = draw_debug(frame, bx, by, theta_lat)

# 結果を表示 or 保存
cv2.imshow('Debug', frame)
cv2.waitKey(0)
cv2.destroyAllWindows()
#cv2.imwrite('debug_output.jpg', frame)

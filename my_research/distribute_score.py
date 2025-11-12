import cv2
import numpy as np
from ultralytics import YOLO
import math

# --- 設定 ---
MODEL_PATH = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans17\weights\best.pt"
TARGET_CLASS_NAME = 'braille block'
# 歪みの影響が分かりやすい屋外の画像を入力に使います
INPUT_PATH = r"C:\Users\ata3357\Desktop\zemi_win\panorama\panorama_img\yolo_001_4.jpg" 
OUTPUT_PATH = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\distribute_undistorted_001.jpg"

# !!!!!! ★★★★★ 重要 ★★★★★ !!!!!!
# 以下の「カメラ行列」と「歪み係数」は、あくまでダミーのサンプル値です。
# 正確な補正を行うには、ご自身のカメラでキャリブレーションを行い、
# そこで得られた実際の値に置き換える必要があります。
# 形式を理解するためのサンプルとしてみてください。
# (fx, fy は焦点距離、cx, cy は画像の中心点に近い値)
camera_matrix = np.array([
    [1000, 0, 960],
    [0, 1000, 540],
    [0, 0, 1]
], dtype=np.float32)
# (k1, k2, p1, p2, k3) 放射方向および接線方向の歪み係数
dist_coeffs = np.array([-0.2, 0.03, 0, 0, 0], dtype=np.float32)
# !!!!!! ★★★★★★★★★★★★★★★★★ !!!!!!


# --- ナビゲーション設定 (変更なし) ---
GUIDANCE_MODE = 'left' 
SAFE_ZONE_LEFT_PERCENT = 15.0
SAFE_ZONE_RIGHT_PERCENT = 45.0
MIN_POINTS_FOR_PATH = 7
FIT_POINTS_COUNT = 10

# モデル読み込み
model = YOLO(MODEL_PATH)

# 入力画像読み込み
img_original = cv2.imread(INPUT_PATH)
if img_original is None:
    raise ValueError("画像が読み込めませんでした")

# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
# ★ ここで画像の歪み補正を実行します ★
# ★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★★
img = cv2.undistort(img_original, camera_matrix, dist_coeffs, None, None)
# ----------------------------------------------------------------

h, w = img.shape[:2]

# ROI範囲 (追従フェーズで使う中央エリア)
roi_left = int(w * (0.5 - 1 / 6))
roi_right = int(w * (0.5 + 1 / 6))
cv2.line(img, (roi_left, 0), (roi_left, h), (255, 0, 0), 2)
cv2.line(img, (roi_right, 0), (roi_right, h), (255, 0, 0), 2)

# ★★★★★ これ以降の処理はすべて歪み補正後の画像 `img` に対して行います ★★★★★

# 検出
results = model(img, verbose=False) # `img` を使う
roi_centers = []
for result in results:
    for box in result.boxes:
        class_id = int(box.cls)
        class_name = model.names[class_id]
        if class_name == TARGET_CLASS_NAME:
            x_center, y_center, _, _ = map(int, box.xywh[0])
            if roi_left <= x_center <= roi_right:
                roi_centers.append([x_center, y_center])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cv2.rectangle(img, (x1, y1), (x2, y2), (255, 255, 0), 2)
                cv2.circle(img, (x_center, y_center), 6, (0, 255, 0), -1)
roi_centers = np.array(roi_centers)


# --- 判定ロジック ---
# (この中のロジックは前回から変更ありません)
status = "none"
angle = None
if roi_centers.shape[0] < MIN_POINTS_FOR_PATH:
    status = "too few points"
else:
    sorted_points = roi_centers[roi_centers[:, 1].argsort()[::-1]]
    points_to_fit = sorted_points[:FIT_POINTS_COUNT]
    vx, vy, x0, y0 = cv2.fitLine(points_to_fit, cv2.DIST_L2, 0, 0.01, 0.01)
    vx, vy, x0, y0 = vx[0], vy[0], x0[0], y0[0]
    angle = math.degrees(math.atan2(vy, vx))
    relative_x = x0
    if abs(vy) > 1e-6:
        relative_x = (vx / vy) * (h - 1 - y0) + x0
    relative_pos_percent = 100 * (relative_x - roi_left) / (roi_right - roi_left)
    if GUIDANCE_MODE == 'left':
        if SAFE_ZONE_LEFT_PERCENT <= relative_pos_percent <= SAFE_ZONE_RIGHT_PERCENT:
            status = "straight (left mode)"
        elif relative_pos_percent < SAFE_ZONE_LEFT_PERCENT:
            status = "left"
        else:
            status = "right"
    elif GUIDANCE_MODE == 'right':
        right_safe_zone_start = 100 - SAFE_ZONE_RIGHT_PERCENT
        right_safe_zone_end = 100 - SAFE_ZONE_LEFT_PERCENT
        if right_safe_zone_start <= relative_pos_percent <= right_safe_zone_end:
            status = "straight (right mode)"
        elif relative_pos_percent > right_safe_zone_end:
            status = "right"
        else:
            status = "left"

# --- ラベル描画 ---
cv2.putText(img, f"Status: {status}", (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
if angle is not None:
    cv2.putText(img, f"Angle: {angle:.1f} deg", (20, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 2)

# --- 直線を描画 ---
if status not in ["none", "too few points"]:
    y1, y2 = 0, h - 1
    if abs(vy) > 1e-6:
        x1 = (vx / vy) * (y1 - y0) + x0
        x2 = (vx / vy) * (y2 - y0) + x0
        cv2.line(img, (int(x1), y1), (int(x2), y2), (0, 0, 255), 3)
    else:
        cv2.line(img, (0, int(y0)), (w-1, int(y0)), (0, 0, 255), 3)

# --- 保存 ---
success = cv2.imwrite(OUTPUT_PATH, img)
if success:
    print(f"保存成功: {OUTPUT_PATH}")
else:
    print("保存失敗。パスを確認してください。")
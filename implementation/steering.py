#!/usr/bin/env python3
import time, math, cv2, numpy as np
from ultralytics import YOLO
from whill import ComWHILL
from sklearn.linear_model import RANSACRegressor, LinearRegression
from collections import deque

# --- WHILL 初期化 ---
whill = ComWHILL(port='COM3')
whill.set_power(True)
time.sleep(1)

# --- パラメータ ---
CAMERA_ID         = 0
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans\weights\best.pt"
YOLO_INPUT_SIZE   = (640, 480)
CONF_THRESH       = 0.2
FOV_DEG           = 360
ROI_ANGLE_DEG     = 60
FORWARD_SPEED_PCT = 20
LOOP_INTERVAL     = 0.1
MAX_WIN_W, MAX_WIN_H = 800, 600

RANSAC_THRESH_PX  = 5.0
INLIER_RATIO_TH   = 0.55
Kp                = 0.1  # 比例ゲイン

# 出力動画設定
OUTPUT_DIR  = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
OUTPUT_PATH = OUTPUT_DIR + "\\course001_angle.mp4"
FPS         = 20.0

# --- 状態履歴 ---
direction_history = deque(maxlen=20)
started = False

def drive_velocity(front_pct, side_pct=0):
    whill.send_velocity(int(front_pct/100*1000),
                        int(side_pct/100*1500))

# --- 点字ブロック検出 ---
model = YOLO(YOLO_MODEL_PATH)

# 角度誤差・インライア比率算出関数
def analyze_geometry(roi_info):
    centers = np.array([[(x1+x2)/2,(y1+y2)/2] for x1,y1,x2,y2 in roi_info])
    if len(centers) < 2:
        return 0.0, 0.0, False
    X = centers[:,0].reshape(-1,1)
    Y = centers[:,1]
    ransac = RANSACRegressor(LinearRegression(),
                             residual_threshold=RANSAC_THRESH_PX,
                             random_state=0)
    ransac.fit(X, Y)
    m = ransac.estimator_.coef_[0]
    theta_rad = math.atan(m)
    angle_err = math.degrees(theta_rad)
    residue = np.abs(m*centers[:,0] - centers[:,1] + ransac.estimator_.intercept_) / math.sqrt(m*m+1)
    inlier_ratio = np.sum(residue < RANSAC_THRESH_PX) / len(residue)
    roi_ok = abs(angle_err) <= ROI_ANGLE_DEG
    return angle_err, inlier_ratio, roi_ok

# カメラ準備
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"カメラ {CAMERA_ID} を開けません。")

fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (fw, fh))

print("Press 's' to start autonomous drive, 'q' to quit.")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret: break

    # YOLO 推論
    small = cv2.resize(frame, YOLO_INPUT_SIZE)
    res   = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()

    # ROI 判定
    sx, sy = frame.shape[1]/YOLO_INPUT_SIZE[0], frame.shape[0]/YOLO_INPUT_SIZE[1]
    roi_info = []
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        theta = (cx - YOLO_INPUT_SIZE[0]/2) / YOLO_INPUT_SIZE[0] * FOV_DEG
        if abs(theta) <= ROI_ANGLE_DEG:
            roi_info.append((int(x1*sx), int(y1*sy), int(x2*sx), int(y2*sy)))

    # キーイベント処理
    disp = frame.copy()
    key = cv2.waitKey(1) & 0xFF
    if not started:
        cv2.putText(disp, "Press 's' to start", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow('WHILL Control', disp)
        if key == ord('s'): started = True
        if key == ord('q'): break
        continue
    if key == ord('q'): break

    # 制御ロジック
    if roi_info:
        angle_err, inlier_ratio, roi_ok = analyze_geometry(roi_info)
        # 比例制御
        if inlier_ratio >= INLIER_RATIO_TH and roi_ok:
            steering = Kp * angle_err
            steering = max(min(steering, 100), -100)
            drive_velocity(FORWARD_SPEED_PCT, steering)
        else:
            drive_velocity(0, 0)
    else:
        drive_velocity(0, 0)

    # 描画
    x1_roi = int((0.5-ROI_ANGLE_DEG/360)*fw)
    x2_roi = int((0.5+ROI_ANGLE_DEG/360)*fw)
    cv2.rectangle(disp, (x1_roi,0), (x2_roi,fh), (255,0,0), 2)
    for x1o,y1o,x2o,y2o in roi_info:
        cv2.rectangle(disp, (x1o,y1o), (x2o,y2o), (0,255,0), 2)
    status = f"Err:{angle_err:.1f}° Inl:{inlier_ratio:.2f}"
    cv2.putText(disp, status, (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    out.write(disp)
    scale = min(MAX_WIN_W/fw, MAX_WIN_H/fh, 1.0)
    if scale < 1.0:
        disp = cv2.resize(disp, (int(fw*scale), int(fh*scale)))
    cv2.imshow('WHILL Control', disp)

    elapsed = time.time() - t0
    if elapsed < LOOP_INTERVAL:
        time.sleep(LOOP_INTERVAL - elapsed)

# 後処理
drive_velocity(0,0)
whill.set_power(False)
cap.release()
out.release()
cv2.destroyAllWindows()

#!/usr/bin/env python3
import time, math, cv2, numpy as np
from ultralytics import YOLO
from whill import ComWHILL
from sklearn.linear_model import RANSACRegressor, LinearRegression

# --- WHILL 初期化 ---
whill = ComWHILL(port='COM3')
whill.set_power(True)
time.sleep(1)

# --- パラメータ ---
CAMERA_ID         = 0
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans\weights\best.pt"
YOLO_INPUT_SIZE   = (640, 480)   # (sw, sh)
CONF_THRESH       = 0.2
FOV_DEG           = 360
ROI_ANGLE_DEG     = 60
SIDE_THRESH_DEG   = 10
CAM_H             = 1.50
ERROR_OFFSET      = 0.0
FORWARD_SPEED_PCT = 30
LOOP_INTERVAL     = 0.1
MAX_WIN_W, MAX_WIN_H = 800, 600

# RANSAC 関連
RANSAC_THRESH_PX  = 5.0
INLIER_RATIO_TH   = 0.6

# 出力動画設定
OUTPUT_DIR  = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
OUTPUT_PATH = OUTPUT_DIR + "\\course018.mp4"
FPS         = 20.0

started = False

def drive_velocity(front_pct, side_pct=0):
    whill.send_velocity(int(front_pct/100*1000),
                        int(side_pct/100*1500))

cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"カメラ {CAMERA_ID} を開けません。")
model = YOLO(YOLO_MODEL_PATH)
sw, sh = YOLO_INPUT_SIZE

fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out    = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (fw, fh))

print("Press 's' to start autonomous drive, 'q' to quit.")

while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        print("ERROR: フレーム取得失敗")
        break

    # 推論
    small = cv2.resize(frame, (sw, sh))
    res   = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()

    # ROI帯座標
    fh_, fw_ = frame.shape[:2]
    sx, sy   = fw_/sw, fh_/sh
    x1_roi = int((0.5 - ROI_ANGLE_DEG/360) * sw * sx)
    x2_roi = int((0.5 + ROI_ANGLE_DEG/360) * sw * sx)

    # ROI検出 & 真横検出情報
    roi_info  = []  # (x1o,y1o,x2o,y2o)
    side_info = []  # (x1o,y1o,x2o,y2o, Dg)
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        # θ計算
        theta = (cx - sw/2) / sw * FOV_DEG

        # 前方ROI
        if abs(theta) <= ROI_ANGLE_DEG:
            roi_info.append((int(x1*sx), int(y1*sy),
                            int(x2*sx), int(y2*sy)))

        # 真横判定＋距離
        if abs(abs(theta) - 90.0) <= SIDE_THRESH_DEG:
            y_max = max(y1, y2, (y1 + y2)/2)
            phi = ((y_max / sh) - 0.5) * math.pi
            if abs(phi) < 1e-3:
                Dg = float('inf')
            else:
                Dg = CAM_H / math.tan(abs(phi)) - ERROR_OFFSET
                Dg = max(Dg, 0.0)
            side_info.append((int(x1*sx), int(y1*sy),
                            int(x2*sx), int(y2*sy),
                            Dg))

    # RANSAC と方向判定
    direction = 'no detection'
    inlier_count = 0
    total_pts    = 0
    masks        = None
    if len(roi_info) >= 2:
        centers = np.array([[(x1+x2)/2, (y1+y2)/2] for x1,y1,x2,y2 in roi_info])
        X = centers[:,0].reshape(-1,1)
        Y = centers[:,1]
        ransac = RANSACRegressor(LinearRegression(),
                                residual_threshold=RANSAC_THRESH_PX,
                                random_state=0)
        ransac.fit(X, Y)
        masks = ransac.inlier_mask_
        inlier_count = masks.sum()
        total_pts    = len(masks)
        inlier_ratio = inlier_count / total_pts
        direction    = 'straight' if inlier_ratio >= INLIER_RATIO_TH else 'turn'

    # 開始前
    disp = frame.copy()
    if not started:
        cv2.putText(disp, "Press 's' to start", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow('WHILL Control', disp)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            started = True
        elif key == ord('q'):
            break
        continue

    # 制御
    drive_velocity(FORWARD_SPEED_PCT if roi_info else 0, 0)

    # 描画
    # ROI枠
    cv2.rectangle(disp, (x1_roi,0), (x2_roi,fh_), (255,0,0), 2)

    # ROI内 Boxes
    if masks is not None:
        for idx, (x1o,y1o,x2o,y2o) in enumerate(roi_info):
            color = (0,255,0) if masks[idx] else (0,165,255)
            cv2.rectangle(disp, (x1o,y1o), (x2o,y2o), color, 2)
    else:
        for x1o,y1o,x2o,y2o in roi_info:
            cv2.rectangle(disp, (x1o,y1o), (x2o,y2o), (0,255,0), 2)

    # 真横 Boxes + 距離表示
    for x1o, y1o, x2o, y2o, Dg in side_info:
        cv2.rectangle(disp, (x1o,y1o), (x2o,y2o), (0,255,255), 2)
        cv2.putText(disp, f"{Dg:.2f}m", (x1o, y1o-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)

    # 状態テキスト
    status = f"{direction}"
    if masks is not None:
        # 少数点2桁で Inlier 比率を表示
        status += f"  Inlier ratio: {inlier_ratio:.2f}"

    cv2.putText(disp, status, (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # 録画・表示
    out.write(disp)
    scale = min(MAX_WIN_W/fw_, MAX_WIN_H/fh_, 1.0)
    if scale < 1.0:
        disp = cv2.resize(disp, (int(fw_*scale), int(fh_*scale)),
                        interpolation=cv2.INTER_AREA)
    cv2.imshow('WHILL Control', disp)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    # ループ周期維持
    dt = time.time() - t0
    if dt < LOOP_INTERVAL:
        time.sleep(LOOP_INTERVAL - dt)

# 後処理
drive_velocity(0,0)
whill.set_power(False)
cap.release()
out.release()
cv2.destroyAllWindows()
print("Program terminated.")

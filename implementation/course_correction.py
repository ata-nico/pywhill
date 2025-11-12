#!/usr/bin/env python3
import time, math, cv2, numpy as np
from ultralytics import YOLO
from whill import ComWHILL

# --- WHILL 初期化 ---
whill = ComWHILL(port='COM3')
whill.send_power_on()
whill.set_battery_voltage_output_mode(vbatt_on_off=True)
time.sleep(1)

# --- パラメータ ---
CAMERA_ID         = 0
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans\weights\best.pt"
YOLO_INPUT_SIZE   = (640, 480)   # (sw, sh)
CONF_THRESH       = 0.2
FOV_DEG           = 360          # 全水平視野角（パノラマ想定）
ROI_ANGLE_DEG     = 60           # ±ROI_ANGLE_DEG° を前方とみなす
SIDE_THRESH_DEG   = 15           # 真横±この角度を側方検出
CAM_H             = 1.55         # カメラ高さ[m]
ERROR_OFFSET      = 0.12         # 誤差補正[m]
FORWARD_SPEED_PCT = 20           # 前進速度[%]
LOOP_INTERVAL     = 0.1          # ループ周期[s]
MAX_WIN_W, MAX_WIN_H = 800, 600  # 表示ウィンドウ最大サイズ
RSS_THRESH        = 800       # RSS閾値（直進/旋回判定）


# --- 出力動画設定 ---
OUTPUT_DIR = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video"
OUTPUT_PATH = OUTPUT_DIR + "\\course002.mp4"
FPS = 20.0

# --- 自動走行開始フラグ ---
started = False

def drive_velocity(front_pct, side_pct=0):
    #WHILLへ前進・側方速度を送信（%→SDK値に変換）
    whill.send_velocity(int(front_pct/100*1000),
                        int(side_pct/100*1500))

# --- カメラ & モデル準備 ---
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"カメラ {CAMERA_ID} を開けませんでした。")
model = YOLO(YOLO_MODEL_PATH)
sw, sh = YOLO_INPUT_SIZE

# フレーム解像度取得
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# VideoWriter初期化
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_PATH, fourcc, FPS, (fw, fh))

print("Press 's' to start autonomous drive, 'q' to quit.")

# --- メインループ ---
while True:
    t0 = time.time()
    ret, frame = cap.read()
    if not ret:
        print("ERROR: フレーム取得失敗")
        break

    # キー受付（開始前）
    key = cv2.waitKey(1) & 0xFF
    if not started:
        cv2.putText(frame, "Press 's' to start", (50,50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
        cv2.imshow('WHILL Control', frame)
        if key == ord('s'):
            started = True
            print("Autonomous drive started.")
        elif key == ord('q'):
            break
        continue

    # 推論用リサイズ & YOLO 推論
    small = cv2.resize(frame, (sw, sh))
    res   = model(small, conf=CONF_THRESH)[0]
    boxes = res.boxes.xyxy.cpu().numpy()

    # ROIおよび真横検出
    roi_boxes = []      # 前方ROI内ボックス
    centers   = []      # RSS用中心座標リスト
    side_info = []      # 側方(Dg推定)
    for x1, y1, x2, y2 in boxes:
        cx = (x1 + x2) / 2
        theta = (cx - sw/2) / sw * FOV_DEG

        # ROI判定
        if abs(theta) <= ROI_ANGLE_DEG:
            roi_boxes.append((int(x1), int(y1), int(x2), int(y2)))
            centers.append(( (x1+x2)/2, (y1+y2)/2 ))

        # 真横判定
        if abs(abs(theta) - 90.0) <= SIDE_THRESH_DEG:
            y_max = max(y1, y2, (y1 + y2)/2)
            phi = ((y_max / sh) - 0.5) * math.pi
            if abs(phi) < 1e-3:
                Dg = float('inf')
            else:
                Dg = CAM_H / math.tan(abs(phi)) - ERROR_OFFSET
                Dg = max(Dg, 0.0)
            side_info.append((int(x1), int(y1), int(x2), int(y2), Dg))

    # RSSによる直進/旋回判定
    direction = 'no detection'
    rss = 0.0
    if len(centers) >= 2:
        pts = np.array(centers)  # shape=(N,2)
        A   = np.vstack([pts[:,0], np.ones(len(pts))]).T
        m, b = np.linalg.lstsq(A, pts[:,1], rcond=None)[0]
        resid = np.abs(m*pts[:,0] - pts[:,1] + b) / math.hypot(m, 1)
        rss   = float(np.sum(resid**2))
        direction = 'straight' if rss < RSS_THRESH else 'turn'

    # 制御: 直進/停止 (ROI 検出があれば直進)
    drive_velocity(FORWARD_SPEED_PCT if roi_boxes else 0, 0)

    # 制御：直進 or 停止
    #if direction == 'straight':
        #drive_velocity(FORWARD_SPEED_PCT, 0)
    #else:
        #drive_velocity(0, 0)

    # 描画スケール算出
    fh, fw = frame.shape[:2]
    sx, sy = fw/sw, fh/sh

    # ROI枠
    roi_x1 = int((0.5 - ROI_ANGLE_DEG/360) * sw * sx)
    roi_x2 = int((0.5 + ROI_ANGLE_DEG/360) * sw * sx)
    cv2.rectangle(frame, (roi_x1, 0), (roi_x2, fh), (255, 0, 0), 2)

    # ROI内緑描画
    for x1, y1, x2, y2 in roi_boxes:
        x1o, y1o = int(x1*sx), int(y1*sy)
        x2o, y2o = int(x2*sx), int(y2*sy)
        cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 255, 0), 2)

    # 側方黄描画＋距離表示
    for x1, y1, x2, y2, Dg in side_info:
        x1o, y1o = int(x1*sx), int(y1*sy)
        x2o, y2o = int(x2*sx), int(y2*sy)
        cv2.rectangle(frame, (x1o, y1o), (x2o, y2o), (0, 255, 255), 3)
        cv2.putText(frame, f"{Dg:.2f}m", (x1o, y1o-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 方向テキスト描画
    cv2.putText(frame, f"Dir:{direction} RSS={int(rss)}", (20,40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    # 録画
    out.write(frame)

    # 表示用リサイズ＆キー監視
    disp = frame.copy()
    h_d, w_d = disp.shape[:2]
    scale = min(MAX_WIN_W/w_d, MAX_WIN_H/h_d, 1.0)
    if scale < 1.0:
        disp = cv2.resize(disp, (int(w_d*scale), int(h_d*scale)),
                        interpolation=cv2.INTER_AREA)
    cv2.imshow('WHILL Control', disp)

    # 終了キー監視
    if key == ord('q'):
        break

    # ループ周期維持
    dt = time.time() - t0
    if dt < LOOP_INTERVAL:
        time.sleep(LOOP_INTERVAL - dt)

# 後処理
cap.release()
cv2.destroyAllWindows()
drive_velocity(0, 0)
whill.send_power_off()
print("Program terminated.")

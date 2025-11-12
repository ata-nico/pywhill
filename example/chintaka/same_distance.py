import cv2
import numpy as np
from ultralytics import YOLO
import math
import time
from whill import ComWHILL

# --- WHILL 初期化 ---
whill = ComWHILL(port='COM3')
whill.set_power(True)
time.sleep(1)

def drive_motor(v_pct, omega_pct):
    whill.send_velocity(int(v_pct/100*1000), int(omega_pct/100*1500))

# --- 設定 ---
CAMERA_ID       = 0
MODEL_PATH      = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt"
ROI_ANGLE_DEG   = 60
FOV_DEG         = 360
CONF_THRESH     = 0.2
SIDE_THRESH_DEG = 10
CAM_H           = 1.55

# 制御パラメータ
Kp              = 100
DIST_THRESHOLD  = 0.05
FORWARD_SPEED   = 30
LOOP_HZ         = 10

# モデルロード
model = YOLO(MODEL_PATH)

# カメラ起動
cap = cv2.VideoCapture(CAMERA_ID)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
interval = 1.0 / LOOP_HZ

# ウィンドウ設定（リサイズ可能に）
cv2.namedWindow("Control View", cv2.WINDOW_NORMAL)
# ウィンドウを400x300ピクセルに縮小
cv2.resizeWindow("Control View", 400, 300)

try:
    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO 推論
        results = model.predict(frame, conf=CONF_THRESH, imgsz=(width, height))
        boxes = results[0].boxes.xyxy.cpu().numpy()

        # 検出ナシ
        if len(boxes) == 0:
            drive_motor(0, 0)
            cv2.putText(frame, "停止：検出なし", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Control View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 各ボックスの theta, Dg を計算
        all_dists = []      # [(theta, Dg, (x1,y1,x2,y2)), ...]
        side_info = []      # 真横用
        theta = None
        for (x1,y1,x2,y2) in boxes:
            cx, cy = (x1+x2)/2, (y1+y2)/2
            theta = ((cx - width/2) / width) * FOV_DEG
            y_max = max(y1,y2,cy)
            phi = ((y_max/height) - 0.5)*math.pi
            Dg = float('inf') if abs(phi)<1e-3 else CAM_H/math.tan(abs(phi))
            all_dists.append((theta, Dg, (int(x1),int(y1),int(x2),int(y2))))
            if abs(abs(theta)-90.0) <= SIDE_THRESH_DEG:
                side_info.append((theta, Dg, (int(x1),int(y1),int(x2),int(y2))))

        # 必要データチェック
        if not all_dists or not side_info or theta is None:
            drive_motor(0, 0)
            cv2.putText(frame, "停止：データ不足", (10,60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)
            cv2.imshow("Control View", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        # 最短距離と対応ボックス
        theta_all, D_all, best_all_box   = min(all_dists, key=lambda x: x[1])
        theta_side, D_side, best_side_box = min(side_info, key=lambda x: x[1])
        error = D_side - D_all

        # 矩形描画：最短距離 → 赤枠，真横距離 → 緑枠
        x1,y1,x2,y2 = best_all_box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,0,255), 2)
        cv2.putText(frame, f"{D_all:.2f}m", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)

        x1,y1,x2,y2 = best_side_box
        cv2.rectangle(frame, (x1,y1), (x2,y2), (0,255,0), 2)
        cv2.putText(frame, f"{D_side:.2f}m", (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

        # 5パターン比例制御
        if theta>0 and error> DIST_THRESHOLD:
            omega =  Kp*error; drive_motor(0, int(omega)); status="右&遠→回転"
        elif theta>0 and error<-DIST_THRESHOLD:
            omega = -Kp*abs(error); drive_motor(0, int(omega)); status="右&近→離反"
        elif theta<0 and error> DIST_THRESHOLD:
            omega = -Kp*error; drive_motor(0, int(omega)); status="左&遠→回転"
        elif theta<0 and error<-DIST_THRESHOLD:
            omega =  Kp*abs(error); drive_motor(0, int(omega)); status="左&近→離反"
        else:
            drive_motor(FORWARD_SPEED,0); status="直進"

        # ステータス表示
        cv2.putText(frame, f"Dall:{D_all:.2f} Dside:{D_side:.2f}", (10,20),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(255,255,255),2)
        cv2.putText(frame, status, (10,40),
                    cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,255),2)

        # 表示＆停止
        cv2.imshow("Control View", frame)
        if cv2.waitKey(1)&0xFF==ord('q'):
            drive_motor(0,0)
            break

        # ループ周期
        dt = time.time()-t0
        if dt<interval:
            time.sleep(interval-dt)

finally:
    cap.release()
    cv2.destroyAllWindows()
    drive_motor(0,0)
    whill.set_power(False)

# corrected_incorze.py
import os
import time
import math
import cv2
import numpy as np
from ultralytics import YOLO
from whill import ComWHILL   # 実機ライブラリ：実行環境に合わせてコメント解除
from sklearn.linear_model import RANSACRegressor, LinearRegression
from collections import deque
from datetime import datetime

# --- WHILL 初期化 ---
whill = None
try:
    # 実機がある場合は以下を有効にしてください
    whill = ComWHILL(port='COM3')
    whill.set_power(True)
    time.sleep(1)
    print("WHILL: 初期化処理をスキップ（実機接続がコメントアウトされています）")
except Exception as e:
    print(f"WHILLへの接続に失敗しました: {e}")
    whill = None

# --- パラメータ設定 ---
CAMERA_ID         = 0
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt"
YOLO_INPUT_SIZE   = (640, 480)  # (width, height)
CONF_THRESH       = 0.2
FOV_DEG           = 360.0
ROI_ANGLE_DEG     = 60.0
SIDE_THRESH_DEG   = 10.0
CAM_H             = 1.55
ERROR_OFFSET      = 0.0
FORWARD_SPEED_PCT = 40
TURN_SPEED_PCT    = 5  # %（側方補正パーセンテージ）。以前は 2 と小さすぎたため実運用向けに調整しておく
ROTATION_SPEED_PCT= 5  # ★★★ 追加：その場旋回時の速度 ★★★
STOP_DISTANCE_M   = 2.0   # ★★★ 追加：インコース旋回を開始する距離(m) ★★★
MIN_MATCH_COUNT = 10           # ★★★ 追加：目標ブロックと判断するための最低一致特徴点数 ★★★
# ANGLE_MATCH_THRESHOLD  = 10.0
SLOPE_THRESHOLD   = 0.1          # ★★★ 追加：旋回完了を判断する直線の傾きのしきい値 ★★★

# ★★★ 新しいパラメータを追加 ★★★
POST_SPIKE_DELAY_S = 2.5  # スパイク検出後、何秒間直進を続けるか(s)

# --- Side Block Spike Detection Parameters (for OUT-COURSE) ---
SIDE_COUNT_HISTORY_SIZE = 20  # Number of frames to average for spike detection
# SPIKE_ABS_THRESHOLD     = 5   # (固定値ルールのため、現在未使用)
# SPIKE_REL_FACTOR        = 3.0 # (固定値ルールのため、現在未使用)


# ▼▼▼【重要】この一行を追加してください ▼▼▼
IN_COURSE_DETECT_RANGE_M = 2.2 # インコースと判断し始める距離(m)

POST_TURN_SEARCH_SPEED = 15 # 旋回後に壁を探すときの前進速度 (%)

# RANSAC パラメータ
RANSAC_THRESH_PX  = 5.0
INLIER_RATIO_TH   = 0.53
FRONT_TH_DEG      = 15.0

# アライメント（初期整列）パラメータ
Kp               = 50.0       # 比例ゲイン（必要に応じて調整）
DIST_THRESHOLD   = 0.05       # [m]（距離誤差のしきい値）

# 旋回停止判定の閾値
IOU_THRESHOLD     = 0.8 # ★★★ 追加：IoUの閾値 ★★★

# --- 移動平均フィルタ ---
WINDOW_SIZE       = 10
dg_history        = deque(maxlen=WINDOW_SIZE)
direction_history = deque(maxlen=WINDOW_SIZE)
side_block_history = deque(maxlen=SIDE_COUNT_HISTORY_SIZE)

# --- ログ・動画の保存設定 ---
LOG_DIR = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\logs\final_integrated"
os.makedirs(LOG_DIR, exist_ok=True)
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# カメラ準備
cap = cv2.VideoCapture(CAMERA_ID)
if not cap.isOpened():
    raise RuntimeError(f"カメラ {CAMERA_ID} を開けませんでした。")
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(os.path.join(LOG_DIR, f"run_{timestamp}.mp4"), fourcc, 10.0, (fw, fh))

# --- YOLOモデルのロード ---
model = YOLO(YOLO_MODEL_PATH)
sw, sh = YOLO_INPUT_SIZE  # sw: width, sh: height

# ヘルパー関数
def clamp(x, a, b):
    return max(a, min(b, x))

def drive_velocity(f_pct, s_pct=0):
    """
    f_pct: 前進速度（%）
    s_pct: 側方/旋回速度（%）
    WHILL API の送信スケールに依存して変換する。
    """
    # 値をクリップ
    f_pct = clamp(f_pct, -100, 100)
    s_pct = clamp(s_pct, -100, 100)
    if whill:
        try:
            # 実際の送信スケールはデバイス仕様に合わせること
            whill.send_velocity(int(f_pct / 100.0 * 1000), int(s_pct / 100.0 * 1500))
        except Exception as e:
            print(f"WHILL 送信エラー: {e}")
    else:
        # デバッグ時はコンソール表示（実機接続無し）
        # print(f"[SIM] drive_velocity: F={f_pct}%, S={s_pct}%")
        pass

# ★★★ ここから追加 ★★★
def calculate_iou(boxA, boxB):
    """
    2つのバウンディングボックスのIntersection over Union (IoU)を計算する。
    ボックスの形式は (x1, y1, x2, y2)。
    """
    # 各ボックスの座標を取得
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    # 重なっている領域の面積を計算 (intersection)
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)

    # 各ボックスの面積を計算
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    
    # IoUを計算 (和集合の面積 = Aの面積 + Bの面積 - 重なりの面積)
    # ゼロ除算を避ける
    unionArea = float(boxAArea + boxBArea - interArea)
    if unionArea == 0:
        return 0
    
    iou = interArea / unionArea
    return iou
# ★★★ ここまで追加 ★★★

# --- 制御フラグの初期化 ---
aligned   = False
started   = False
First_Dg  = None
lower_thr = upper_thr = None
spike_detection_time = None 

print("'s'キーで開始、'q'キーで終了します。")

try:
    course_type = "STRAIGHT"
    turn_direction = "NONE"
    while True:
        # --- ループごとの状態変数をここでまとめて初期化 ---
        stop_flag = False
        # course_type = "STRAIGHT"
        # turn_direction = "NONE"
        direction = 'straight'
        inlier_ratio = 0.0
        avg_Dg = 0.0
        front_block_for_turn = None
        is_spike = False

        ret, frame = cap.read()
        if not ret:
            print("カメラからフレームを取得できませんでした。")
            break
        disp = frame.copy()

        # キー入力
        key = cv2.waitKey(1) & 0xFF
        if not started:
            cv2.putText(disp, "Press 's' to start", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
            cv2.imshow('Control', disp)
            if key == ord('s'):
                started = True
            elif key == ord('q'):
                break
            continue
        if key == ord('q'):
            break

        # --- 画像処理と物体検出 ---
        h, w = disp.shape[:2]
        scale_x = w / sw
        scale_y = h / sh

        small = cv2.resize(frame, (sw, sh))
        res = model(small, conf=CONF_THRESH)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else np.empty((0,4))

        # --- 検出ブロックの分類 ---
        all_info = []
        side_info = []
        roi_info  = []
        front_info = []

        if len(boxes) > 0:
            for (x1, y1, x2, y2) in boxes:
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                theta = (cx - sw / 2.0) / sw * FOV_DEG
                y_max = max(y1, y2, cy)
                phi = ((y_max / sh) - 0.5) * math.pi
                if abs(math.tan(abs(phi))) < 1e-6:
                    Dg = float('inf')
                else:
                    Dg = CAM_H / math.tan(abs(phi)) - ERROR_OFFSET

                rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
                rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
                
                block_data = (theta, Dg, rx1, ry1, rx2, ry2)
                all_info.append(block_data)

                if abs(theta) <= ROI_ANGLE_DEG:
                    roi_info.append(block_data)
                if abs(abs(theta) - 90.0) <= SIDE_THRESH_DEG:
                    side_info.append(block_data)
                if abs(theta) <= FRONT_TH_DEG:
                    front_info.append(block_data)
                    # --- 検出ブロックの分類 ---
            
            # ▼▼▼ ここから3行追加 ▼▼▼
            front_dist_str = "Front: N/A" # 表示用の文字列を初期化
            if front_info:
                closest_front_dist = min(front_info, key=lambda b: b[1])[1]
                front_dist_str = f"Front: {closest_front_dist:.2f}m"
            # ▲▲▲ ここまで追加 ▲▲▲
        
        # --- 制御ロジック (描画より先に実行) ---
        if len(boxes) > 0 and aligned:
            if not stop_flag and spike_detection_time is None:
                if First_Dg is None:
                    if side_info:
                        First_Dg = min(b[1] for b in side_info)
                        print(f"最初の基準距離を設定: {First_Dg:.2f}m")
                        lower_thr = First_Dg - 0.15
                        upper_thr = First_Dg + 0.15
                else:
                    if side_info:
                        current_side_count = len(side_info)
                        side_block_history.append(current_side_count)

                        if len(side_info) >= 5:
                            X = np.array([b[0] for b in side_info]).reshape(-1, 1)
                            y = np.array([b[1] for b in side_info])
                            ransac = RANSACRegressor(LinearRegression(), residual_threshold=RANSAC_THRESH_PX)
                            try:
                                ransac.fit(X, y)
                                inlier_mask = ransac.inlier_mask_
                                inlier_ratio = np.sum(inlier_mask) / len(X)
                                if inlier_ratio < INLIER_RATIO_TH:
                                    direction = 'turn'
                            except ValueError:
                                direction = 'turn'
                        elif len(side_info) > 0:
                            direction = 'turn'

                        # ★★★ ここから旋回方向の判定ロジックを全面的に変更 ★★★
                        if direction == 'turn':
                            # --- Step 1: まずインコースかアウトコースかを確定させる ---
                            course_type = "OUT-COURSE"  # デフォルトはアウトコース
                            front_block_candidate = None
                            if roi_info:
                                # 正面に最も近いブロックを探す
                                front_block_candidate_tuple = min(roi_info, key=lambda x: abs(x[0]))
                                front_block_candidate = {'theta': front_block_candidate_tuple[0], 'dist': front_block_candidate_tuple[1]}

                            # 正面ブロックがあり、一定の距離内ならインコースと判断
                            if front_block_candidate is not None and abs(front_block_candidate['theta']) <= FRONT_TH_DEG:
                                d_front = front_block_candidate['dist']
                                if d_front <= IN_COURSE_DETECT_RANGE_M:
                                    course_type = "IN-COURSE"
                                    front_block_for_turn = front_block_candidate # インコース旋回用に情報を保持
                            
                            # --- Step 2: コースタイプに応じて、旋回方向のルールを切り替える ---
                            if side_info:
                                avg_side_theta = np.mean([b[0] for b in side_info])
                                is_wall_on_left = (avg_side_theta < 0)

                                if course_type == "IN-COURSE":
                                    # インコースの場合：壁側へ曲がる（内回り）
                                    # 壁が左にあれば -> 右旋回 (時計回り)
                                    turn_direction = "RIGHT TURN" if is_wall_on_left else "LEFT TURN"
                                else: # OUT-COURSE
                                    # アウトコースの場合：壁から離れるように曲がる（外回り）
                                    # 壁が左にあれば -> 左旋回 (反時計回り)
                                    turn_direction = "LEFT TURN" if is_wall_on_left else "RIGHT TURN"

                        # --- Step 3: 停止フラグを立てる ---
                            should_stop_for_turn = False
                            if course_type == "IN-COURSE":
                                # インコースの場合：front_block_for_turnに格納された距離をチェックし、
                                # STOP_DISTANCE_M より近い場合にのみ停止準備に入る
                                if front_block_for_turn is not None and front_block_for_turn['dist'] <= STOP_DISTANCE_M:
                                    should_stop_for_turn = True
                                    print(f"インコース目標が停止距離内 ( <= {STOP_DISTANCE_M}m) に入りました。")
                            
                            elif course_type == "OUT-COURSE":
                                # アウトコースの場合：スパイク(ブロック4個以上)を検出した場合のみ停止準備に入る
                                is_spike = (current_side_count >= 4)
                                if is_spike:
                                    should_stop_for_turn = True

                            # 停止準備フラグがTrueに設定され、かつタイマーが作動していない場合に遅延を開始
                            if should_stop_for_turn and spike_detection_time is None:
                                print(f"{course_type} を検出！ {POST_SPIKE_DELAY_S}秒の遅延を開始します。")
                                spike_detection_time = time.time()
                        # ★★★ 旋回方向ロジックの変更ここまで ★★★

        # --- 描画処理 ---
        x_left_thresh  = int((0.5 - FRONT_TH_DEG / FOV_DEG) * w)
        x_right_thresh = int((0.5 + FRONT_TH_DEG / FOV_DEG) * w)
        cv2.line(disp, (x_left_thresh, 0), (x_left_thresh, h), (255, 0, 255), 1)
        cv2.line(disp, (x_right_thresh, 0), (x_right_thresh, h), (255, 0, 255), 1)

        x_roi_left  = int((0.5 - ROI_ANGLE_DEG / FOV_DEG) * w)
        x_roi_right = int((0.5 + ROI_ANGLE_DEG / FOV_DEG) * w)
        cv2.line(disp, (x_roi_left, 0), (x_roi_left, h), (255, 255, 0), 1)
        cv2.line(disp, (x_roi_right, 0), (x_roi_right, h), (255, 255, 0), 1)

        for _, _, x1, y1, x2, y2 in roi_info:
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 255, 255), 1)
        for _, _, x1, y1, x2, y2 in front_info:
            cv2.rectangle(disp, (x1, y1), (x2, y2), (255, 0, 255), 1)

        if is_spike:
            for _, _, x1, y1, x2, y2 in side_info:
                cv2.rectangle(disp, (x1, y1), (x2, y2), (0, 0, 255), 2)
        else:
            if side_info:
                _, _, sx1, sy1, sx2, sy2 = min(side_info, key=lambda x: x[1])
                cv2.rectangle(disp, (sx1, sy1), (sx2, sy2), (0, 255, 0), 2)

        if all_info:
            _, closest_block_dist, ax1, ay1, ax2, ay2 = min(all_info, key=lambda x: x[1])
            cv2.rectangle(disp, (ax1, ay1), (ax2, ay2), (255, 0, 0), 2)
            cv2.putText(disp, f"Closest: {closest_block_dist:.2f}m", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
                # 正面ブロックの距離情報を左上に表示 (色は紫)
        cv2.putText(disp, front_dist_str, (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
        
# --- 走行・停止制御 ---
        if stop_flag:
            drive_velocity(0, 0)
        elif not aligned:
            # アライメントロジック
            if not side_info:
                # 起動直後などで横ブロックが見えない場合は、その場でゆっくり旋回して探す
                drive_velocity(0, -TURN_SPEED_PCT)
            else:
                _, D_all, _, _, _, _ = min(all_info, key=lambda x: x[1]); _, D_side, _, _, _, _ = min(side_info, key=lambda x: x[1])
                error = D_side - D_all
                if error > DIST_THRESHOLD: drive_velocity(0, -TURN_SPEED_PCT)
                elif error < -DIST_THRESHOLD: drive_velocity(0, TURN_SPEED_PCT)
                else: drive_velocity(0, 0); aligned = True
        elif spike_detection_time is not None:
            # スパイク検出後の遅延処理
            elapsed = time.time() - spike_detection_time
            if elapsed >= POST_SPIKE_DELAY_S:
                stop_flag = True
                print(f"遅延完了 ({POST_SPIKE_DELAY_S}秒)。旋回のため停止します。")
                drive_velocity(0, 0)
            else:
                drive_velocity(FORWARD_SPEED_PCT, 0)
        # ★★★ ここからが新しい壁追従ロジック ★★★
        elif First_Dg is not None:
            # 基準距離が設定されていれば、壁追従モードに入る
            if side_info:
                # 側面ブロックが見える場合：通常の壁追従
                dg = min(side_info, key=lambda b: b[1])[1]
                forward, side = FORWARD_SPEED_PCT, 0
                if dg < lower_thr: side = TURN_SPEED_PCT
                elif dg > upper_thr: side = -TURN_SPEED_PCT
                drive_velocity(forward, side)
            else:
                # 側面ブロックが見えない場合：ゆっくり前進して探す
                print("壁追従中、側面ブロックを探索しています...")
                drive_velocity(15, 0)
        else:
            # 上記のいずれでもない場合（起動直後など）は停止
            drive_velocity(0, 0)
            if len(boxes) == 0: cv2.putText(disp, "No detection", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # --- 旋回実行フェーズ ---
        if stop_flag:
            if course_type == "IN-COURSE" and front_block_for_turn is not None:
                # (インコースの旋回ロジックは変更なし)
                TIME_PER_DEGREE = 0.132; TURN_SPEED_FOR_CALIBRATION = 5
                print("α+δの角度推定と時間制御による旋回を開始します。"); cv2.waitKey(1500)
                ret, stable_frame = cap.read(); disp = stable_frame.copy(); target_angle = 90.0
                if ret:
                    small_stable = cv2.resize(stable_frame, (sw, sh)); res_stable = model(small_stable, conf=CONF_THRESH)[0]
                    boxes_stable = res_stable.boxes.xyxy.cpu().numpy(); front_blocks, side_blocks = [], []
                    if len(boxes_stable) > 0:
                        for x1,y1,x2,y2 in boxes_stable:
                            cx_s, cy_s = (x1+x2)/2.0, (y1+y2)/2.0; theta_s = (cx_s - sw/2.0)/sw * FOV_DEG
                            rx1, ry1, rx2, ry2 = int(x1*scale_x), int(y1*scale_y), int(x2*scale_x), int(y2*scale_y)
                            center_point = np.array([(rx1+rx2)/2, (ry1+ry2)/2])
                            block_info = {'box': (rx1, ry1, rx2, ry2), 'center': center_point}
                            if abs(theta_s) <= FRONT_TH_DEG: front_blocks.append(block_info)
                            is_valid_side = (turn_direction == "RIGHT TURN" and theta_s < 0) or \
                                          (turn_direction == "LEFT TURN" and theta_s > 0)
                            if is_valid_side: side_blocks.append(block_info)
                    angle_alpha, angle_delta = 0.0, 0.0
                    if front_blocks and side_blocks:
                        sorted_front_blocks = sorted(front_blocks, key=lambda b: b['center'][1], reverse=True)
                        found_valid_angle = False
                        for b_candidate in sorted_front_blocks:
                            point_B = b_candidate['center']
                            c_candidate = min(side_blocks, key=lambda c: np.linalg.norm(c['center'] - point_B))
                            point_C = c_candidate['center']
                            if np.linalg.norm(point_B - point_C) < 50: continue
                            camera_A = np.array([w / 2, h]); vec_AB = point_B - camera_A; vec_AC = point_C - camera_A
                            vec_CB = point_B - point_C; vec_CA = camera_A - point_C
                            dot_d = np.dot(vec_AB, vec_AC); mag_AB = np.linalg.norm(vec_AB); mag_AC = np.linalg.norm(vec_AC)
                            angle_delta = np.degrees(np.arccos(clamp(dot_d / (mag_AB * mag_AC), -1.0, 1.0))) if mag_AB > 0 and mag_AC > 0 else 0
                            dot_a = np.dot(vec_CB, vec_CA); mag_CB = np.linalg.norm(vec_CB); mag_CA = np.linalg.norm(vec_CA)
                            angle_alpha = np.degrees(np.arccos(clamp(dot_a / (mag_CB * mag_CA), -1.0, 1.0))) if mag_CB > 0 and mag_CA > 0 else 0
                            total_angle = angle_alpha + angle_delta
                            if total_angle > 10.0:
                                target_angle = total_angle; found_valid_angle = True
                                print(f"有効な角度を検出しました: B={point_B.astype(int)}, C={point_C.astype(int)}"); break
                        if not found_valid_angle:
                            print("警告: 全ての候補ペアで有効な角度を推定できませんでした。90度で旋回します。")
                            target_angle = 90.0; angle_alpha, angle_delta = 0, 0
                        cv2.putText(disp, f"Alpha (a): {angle_alpha:.1f} deg", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(disp, f"Delta (d): {angle_delta:.1f} deg", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        cv2.putText(disp, f"TARGET (a+d): {target_angle:.1f} deg", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    else: print("角度推定に必要なブロックが見つかりませんでした。デフォルトの90度で旋回します。")
                    cv2.imshow('Control', disp); cv2.waitKey(2000)

                turn_duration = target_angle * TIME_PER_DEGREE
                start_time = time.time()
                while time.time() - start_time < turn_duration:
                    ret, turn_frame = cap.read()
                    if not ret: break
                    cv2.putText(turn_frame, f"Turning... Target: {target_angle:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Control', turn_frame); out.write(turn_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): print("手動で旋回を中断しました。"); break
                    drive_velocity(0, TURN_SPEED_FOR_CALIBRATION if turn_direction == "RIGHT TURN" else -TURN_SPEED_FOR_CALIBRATION)
                drive_velocity(0, 0); print("旋回完了。"); cv2.waitKey(1500)
                
                print("走行再開のため、新しい基準距離を計測します。")
                ret, new_frame = cap.read()
                if ret:
                    out.write(new_frame)
                    small_new = cv2.resize(new_frame, (sw, sh)); res_new = model(small_new, conf=CONF_THRESH)[0]
                    boxes_new = res_new.boxes.xyxy.cpu().numpy(); side_blocks_new = []
                    if len(boxes_new) > 0:
                        for x1,y1,x2,y2 in boxes_new:
                            cx_n, cy_n = (x1+x2)/2.0, (y1+y2)/2.0; theta_n = (cx_n - sw/2.0)/sw * FOV_DEG
                            is_new_side = (turn_direction == "RIGHT TURN" and theta_n < 0) or \
                                          (turn_direction == "LEFT TURN" and theta_n > 0)
                            if is_new_side and abs(abs(theta_n)-90.0) <= SIDE_THRESH_DEG:
                                y_max_n = max(y1,y2,cy_n); phi_n = ((y_max_n / sh) - 0.5) * math.pi
                                Dg_n = CAM_H / math.tan(abs(phi_n)) - ERROR_OFFSET if abs(math.tan(abs(phi_n))) > 1e-6 else float('inf')
                                side_blocks_new.append({'dist': Dg_n})
                    if side_blocks_new:
                        new_first_dg = min(side_blocks_new, key=lambda b: b['dist'])['dist']
                        if new_first_dg != float('inf'):
                            First_Dg = new_first_dg; print(f"新しい基準距離(First_Dg)を更新しました: {First_Dg:.2f}m")
                            dg_history.clear(); direction_history.clear()
                            lower_thr = First_Dg - 0.15; upper_thr = First_Dg + 0.15
                            print(f"新しい閾値=[{lower_thr:.2f},{upper_thr:.2f}]")
                    else:
                        print("警告: 走行再開のための基準ブロックが見つかりませんでした。基準をリセットします。"); First_Dg = None

            elif course_type == "OUT-COURSE":
                TIME_PER_DEGREE = 0.132; TURN_SPEED_FOR_CALIBRATION = 5; TARGET_ANGLE = 90.0
                print(f"アウトコース：固定{TARGET_ANGLE}度の旋回を開始します。"); cv2.waitKey(1500)
                turn_duration = TARGET_ANGLE * TIME_PER_DEGREE
                start_time = time.time()
                while time.time() - start_time < turn_duration:
                    ret, turn_frame = cap.read()
                    if not ret: break
                    cv2.putText(turn_frame, f"Turning OUT-COURSE... Target: {TARGET_ANGLE:.1f} deg", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.imshow('Control', turn_frame); out.write(turn_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                    drive_velocity(0, TURN_SPEED_FOR_CALIBRATION if turn_direction == "RIGHT TURN" else -TURN_SPEED_FOR_CALIBRATION)
                drive_velocity(0, 0); print("旋回完了。"); cv2.waitKey(1500)

                print("新しい壁が見つかるまで、ゆっくり前進します..."); search_start_time = time.time(); found_wall = False
                while True:
                    if time.time() - search_start_time > 60.0: print("タイムアウト: 新しい壁が見つかりませんでした。"); break
                    drive_velocity(POST_TURN_SEARCH_SPEED, 0); ret, search_frame = cap.read()
                    if not ret: break
                    small_search = cv2.resize(search_frame, (sw, sh)); res_search = model(small_search, conf=CONF_THRESH)[0]
                    boxes_search = res_search.boxes.xyxy.cpu().numpy(); side_blocks_new = []
                    if len(boxes_search) > 0:
                        for x1,y1,x2,y2 in boxes_search:
                            cx_n, cy_n = (x1+x2)/2.0, (y1+y2)/2.0; theta_n = (cx_n - sw/2.0)/sw * FOV_DEG
                            is_new_side = (turn_direction == "RIGHT TURN" and theta_n < 0) or \
                                          (turn_direction == "LEFT TURN" and theta_n > 0)
                            if is_new_side and abs(abs(theta_n)-90.0) <= SIDE_THRESH_DEG:
                                y_max_n = max(y1,y2,cy_n); phi_n = ((y_max_n / sh) - 0.5) * math.pi
                                Dg_n = CAM_H / math.tan(abs(phi_n)) - ERROR_OFFSET if abs(math.tan(abs(phi_n))) > 1e-6 else float('inf')
                                side_blocks_new.append({'dist': Dg_n})
                    cv2.putText(search_frame, "Searching for new wall...", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                    cv2.imshow('Control', search_frame); out.write(search_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'): break
                                        # ★★★ ここから下のロジックを変更 ★★★
                    if side_blocks_new:
                        num_side_blocks = len(side_blocks_new)
                        # 新しい終了条件：横ブロックが1個か2個の安定した状態になったら終了
                        if num_side_blocks == 1 or num_side_blocks == 2:
                            drive_velocity(0, 0)
                            print(f"安定した壁を発見しました。(ブロック数: {num_side_blocks})")
                            found_wall = True
                            break # ループを抜けてメインフェーズへ
                        else:
                            # 壁は見えているが、まだ多すぎる（スパイクの密集地帯）ので直進を続ける
                            print(f"壁を検出中...まだブロックが多すぎます。(ブロック数: {num_side_blocks})")
                    # (side_blocks_new が空の場合は、何もせずループが継続され、探索が続く)

                if found_wall:
                    new_first_dg = min(side_blocks_new, key=lambda b: b['dist'])['dist']
                    if new_first_dg != float('inf'):
                        First_Dg = new_first_dg; print(f"新しい基準距離(First_Dg)を更新しました: {First_Dg:.2f}m")
                        dg_history.clear(); direction_history.clear()
                        lower_thr = First_Dg - 0.15; upper_thr = First_Dg + 0.15
                        print(f"新しい閾値=[{lower_thr:.2f},{upper_thr:.2f}]")
                else:
                    print("警告: 走行再開のための基準ブロックが見つかりませんでした。基準をリセットします。"); First_Dg = None
            
            course_type = "STRAIGHT"
            turn_direction = "NONE"
            # 状態をリセットして、壁追従モードに戻す
            stop_flag = False
            spike_detection_time = None
            
            

        # --- 映像保存 & 表示 ---
        status_font = cv2.FONT_HERSHEY_SIMPLEX; status_scale = 0.7; status_thickness = 2
        status_x = w - 350 if w > 400 else 10
        info_color = (255, 255, 0); alert_color = (0, 0, 255)
        cv2.putText(disp, f"Course: {course_type}", (status_x, 30), status_font, status_scale, info_color, status_thickness)
        cv2.putText(disp, f"Path: {direction.upper()}", (status_x, 60), status_font, status_scale, info_color, status_thickness)
        if is_spike:
            cv2.putText(disp, "SPIKE DETECTED!", (status_x, 90), cv2.FONT_HERSHEY_TRIPLEX, 0.8, alert_color, status_thickness)
        out.write(disp)
        cv2.imshow('Control', disp)

finally:
    # 終了処理（必ず実行）
    print("プログラムを終了します...")
    drive_velocity(0, 0)
    if whill:
        try: whill.set_power(False)
        except Exception as e: print(f"WHILL 電源オフエラー: {e}")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("処理が完了し、動画が保存されました。")
import os
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tqdm import tqdm

# --- ユーザー設定項目 ---
VIDEO_PATH        = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video\R0010523_st.MP4"  # ★★★ 解析したいビデオファイルのパスに書き換えてください ★★★
YOLO_MODEL_PATH   = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans20\weights\best.pt"
OUTPUT_VIDEO_PATH = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\logs\final_integrated\side_analysis_video.mp4" # 解析結果のビデオの保存先
OUTPUT_GRAPH_PATH = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\logs\final_integrated\side_block_count_graph.png" # グラフの保存先

YOLO_INPUT_SIZE   = (640, 480)  # (width, height)
CONF_THRESH       = 0.2
FOV_DEG           = 360.0
SIDE_THRESH_DEG   = 5.0 # ±90°からどのくらいの角度を許容するか

# --- 初期化 ---
# YOLOモデルのロード
model = YOLO(YOLO_MODEL_PATH)
sw, sh = YOLO_INPUT_SIZE

# 入力ビデオの準備
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise RuntimeError(f"ビデオファイルを開けませんでした: {VIDEO_PATH}")

# ビデオのプロパティを取得
fw = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
fh = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

# 出力ビデオの準備
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (fw, fh))

# グラフ用のデータ保存リスト
side_block_counts = []

print("ビデオ解析を開始します...")

# --- メイン処理ループ ---
try:
    for _ in tqdm(range(total_frames), desc="Processing Video"):
        ret, frame = cap.read()
        if not ret:
            break

        # YOLOによる物体検出
        small = cv2.resize(frame, (sw, sh))
        res = model(small, conf=CONF_THRESH)[0]
        boxes = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else np.empty((0,4))

        # 描画用のフレームをコピー
        disp = frame.copy()
        h, w = disp.shape[:2]
        scale_x = w / sw
        scale_y = h / sh
        
        # このフレームで検出されたサイドブロックの数をカウント
        current_frame_side_count = 0

        if len(boxes) > 0:
            for (x1, y1, x2, y2) in boxes:
                cx = (x1 + x2) / 2.0
                theta = (cx - sw / 2.0) / sw * FOV_DEG

                # 角度が±90°の範囲内かチェック
                if abs(abs(theta) - 90.0) <= SIDE_THRESH_DEG:
                    current_frame_side_count += 1
                    
                    # 検出したブロックのバウンディングボックスを描画
                    rx1, ry1 = int(x1 * scale_x), int(y1 * scale_y)
                    rx2, ry2 = int(x2 * scale_x), int(y2 * scale_y)
                    cv2.rectangle(disp, (rx1, ry1), (rx2, ry2), (0, 255, 0), 2)
        
        # カウント結果をリストに追加
        side_block_counts.append(current_frame_side_count)

        # --- 描画処理 ---
        # ±90°のおおよその位置にラインを描画 (360°カメラを平面展開した映像を想定)
        x_left_90 = int(w * 0.25)
        x_right_90 = int(w * 0.75)
        cv2.line(disp, (x_left_90, 0), (x_left_90, h), (255, 0, 0), 2) # 左90° (青)
        cv2.line(disp, (x_right_90, 0), (x_right_90, h), (0, 0, 255), 2) # 右90° (赤)

        # 現在のフレームのカウント数を画面に表示
        cv2.putText(disp, f"Side Blocks: {current_frame_side_count}", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

        # 出力ビデオにフレームを書き込み
        out.write(disp)
        
        # (任意) 処理中に映像を表示したい場合は以下のコメントを解除
        # cv2.imshow("Analysis", disp)
        # if cv2.waitKey(1) & 0xFF == ord('q'):
        #     break

finally:
    # 終了処理
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print(f"\nビデオ解析が完了しました。結果は {OUTPUT_VIDEO_PATH} に保存されました。")


# --- グラフ生成 ---
try:
    print("グラフを生成しています...")
    plt.style.use('seaborn-v0_8-whitegrid')
    fig, ax = plt.subplots(figsize=(15, 7))

    ax.plot(side_block_counts, label='Detected Side Blocks', color='royalblue')
    
    # y軸を整数のみに設定
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    
    # ラベルとタイトル
    ax.set_xlabel("Frame Number", fontsize=12)
    ax.set_ylabel("Number of Detected Blocks", fontsize=12)
    ax.set_title("Side Block Count per Frame (at approx. +/- 90 degrees)", fontsize=16)
    ax.legend()
    ax.margins(x=0.01) # x軸の余白を調整
    
    # グラフを画像として保存
    plt.savefig(OUTPUT_GRAPH_PATH)
    print(f"グラフが正常に生成されました。結果は {OUTPUT_GRAPH_PATH} に保存されました。")

except Exception as e:
    print(f"グラフの生成中にエラーが発生しました: {e}")
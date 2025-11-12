import os
import cv2
import math
import numpy as np
import pandas as pd
from ultralytics import YOLO
from sklearn.metrics import f1_score
from sklearn.linear_model import LinearRegression, RANSACRegressor

# ────────── 設定 ──────────
FRAME_DIR       = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\video\inlier"
MODEL_PATH      = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans18\weights\best.pt"  # 実際のbest.ptへのパスに置き換え

# RANSACおよび初期閾値設定
RANSAC_THRESH_PX= 5.0

# YOLOモデルロード
model = YOLO(MODEL_PATH)

# 重心検出
def detect_centers(image_path):
    img = cv2.imread(image_path)
    results = model(img)[0]
    centers = []
    for box in results.boxes.xyxy:
        x1, y1, x2, y2 = box.cpu().numpy()
        centers.append(((x1 + x2) / 2, (y1 + y2) / 2))
    return centers

# RSS計算
def compute_rss(centers):
    if len(centers) < 2:
        return 0.0
    data = np.array(centers)
    X = np.vstack([data[:,0], np.ones(len(data))]).T
    Y = data[:,1]
    m, b = np.linalg.lstsq(X, Y, rcond=None)[0]
    distances = np.abs(m*data[:,0] - data[:,1] + b) / math.sqrt(m*m + 1)
    return float((distances**2).sum())

# インライア比率計算
def compute_inlier_ratio(centers):
    if len(centers) < 2:
        return 0.0
    data = np.array(centers)
    X = data[:,0].reshape(-1,1)
    Y = data[:,1]
    ransac = RANSACRegressor(LinearRegression(), residual_threshold=RANSAC_THRESH_PX, random_state=0)
    ransac.fit(X, Y)
    mask = ransac.inlier_mask_
    return float(mask.sum() / len(data))

# グリッドサーチ
def grid_search(values, labels, name, steps=101):
    best = {'thr': None, 'f1': -1}
    for thr in np.linspace(values.min(), values.max(), steps):
        preds = (values > thr).astype(int)
        score = f1_score(labels, preds)
        if score > best['f1']:
            best = {'thr': thr, 'f1': score}
    print(f"{name} - Best thr: {best['thr']:.3f}, F1: {best['f1']:.3f}")
    return best

# メイン処理
def main():
    records = []
    for fn in os.listdir(FRAME_DIR):
        name, ext = os.path.splitext(fn)
        parts = name.split("_")
        if len(parts) != 3:
            continue
        label_str = parts[2]
        label = 1 if label_str == 'straight' else 0
        path = os.path.join(FRAME_DIR, fn)
        centers = detect_centers(path)
        rss = compute_rss(centers)
        inlier = compute_inlier_ratio(centers)
        records.append({'label': label, 'rss': rss, 'inlier': inlier})

    # DataFrame化
    df = pd.DataFrame(records)

    # 値がすべて同じ場合のエラー回避を適用
    df_rss = df[df.rss > df.rss.min()]
    df_in = df[df.inlier > df.inlier.min()]

    # 各フィルタ後に対応するラベルを取得してグリッドサーチ
    print("--- RSS Grid Search ---")
    rss_labels = df_rss['label'].values
    rss_values = df_rss['rss'].values
    grid_search(rss_values, rss_labels, 'RSS')

    print("--- Inlier Ratio Grid Search ---")
    in_labels = df_in['label'].values
    in_values = df_in['inlier'].values
    grid_search(in_values, in_labels, 'Inlier Ratio')

if __name__ == '__main__':
    main()


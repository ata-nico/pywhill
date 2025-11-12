import os
import numpy as np
import pandas as pd
from sklearn.metrics import (
    roc_curve, auc,
    precision_recall_curve, average_precision_score,
    f1_score, precision_score, recall_score
)
import matplotlib.pyplot as plt

# --- 設定 ---
CSV_PATH       = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\frame\score_results.csv"
OUTPUT_DIR     = os.path.dirname(CSV_PATH)  # 同じフォルダに保存
ROC_PLOT_FILE  = os.path.join(OUTPUT_DIR, "roc_curve.png")
PR_PLOT_FILE   = os.path.join(OUTPUT_DIR, "pr_curve.png")

# --- データ読み込み ---
df = pd.read_csv(CSV_PATH)
y_true = (df['gt_label'] == 'straight').astype(int)

# RANSAC
y_score_ransac = df['inlier_ratio'].values

# RSS（inf を除外）
mask = np.isfinite(df['rss'].values)
y_true_rss = y_true[mask]
y_score_rss = -df['rss'].values[mask]

# --- ROC/AUC 計算 ---
fpr_r, tpr_r, _ = roc_curve(y_true, y_score_ransac)
roc_auc_r       = auc(fpr_r, tpr_r)
fpr_s, tpr_s, _ = roc_curve(y_true_rss, y_score_rss)
roc_auc_s       = auc(fpr_s, tpr_s)

# --- PR/AP 計算 ---
prec_r, rec_r, _ = precision_recall_curve(y_true, y_score_ransac)
ap_r             = average_precision_score(y_true, y_score_ransac)
prec_s, rec_s, _ = precision_recall_curve(y_true_rss, y_score_rss)
ap_s             = average_precision_score(y_true_rss, y_score_rss)

# --- ROCプロット ---
plt.figure()
plt.plot(fpr_r, tpr_r, label=f"RANSAC ROC (AUC={roc_auc_r:.2f})")
plt.plot(fpr_s, tpr_s, label=f"RSS ROC (AUC={roc_auc_s:.2f})")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend()
plt.grid(True)
plt.savefig(ROC_PLOT_FILE, dpi=300)  # ここで保存
plt.close()

# --- PRプロット ---
plt.figure()
plt.plot(rec_r, prec_r, label=f"RANSAC PR (AP={ap_r:.2f})")
plt.plot(rec_s, prec_s, label=f"RSS PR (AP={ap_s:.2f})")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.title("Precision-Recall Curve")
plt.legend()
plt.grid(True)
plt.savefig(PR_PLOT_FILE, dpi=300)   # ここで保存
plt.close()

print(f"ROC curve saved to: {ROC_PLOT_FILE}")
print(f"PR curve saved to : {PR_PLOT_FILE}")

# --- ベスト閾値での F1 比較 ---
# 閾値候補
ths_r = np.linspace(0, 1, 101)
ths_s = np.linspace(y_score_rss.min(), y_score_rss.max(), 101)

results = []
for name, scores, y_t, ths in [
    ("RANSAC", y_score_ransac, y_true, ths_r),
    ("RSS",    y_score_rss,    y_true_rss, ths_s)
]:
    best_f1, best_th = 0, None
    for th in ths:
        preds = (scores >= th).astype(int)
        f1 = f1_score(y_t, preds)
        if f1 > best_f1:
            best_f1, best_th = f1, th
    prec = precision_score(y_t, (scores >= best_th).astype(int))
    rec  = recall_score(y_t,  (scores >= best_th).astype(int))
    acc  = (preds == y_t).mean()
    results.append((name, best_th, acc, prec, rec, best_f1))

print("Method | BestThr |  Acc  | Prec  | Rec   |  F1")
for name, th, acc, prec, rec, f1 in results:
    print(f"{name:6s} | {th:7.3f} | {acc:5.2f} | {prec:5.2f} | {rec:5.2f} | {f1:5.2f}")

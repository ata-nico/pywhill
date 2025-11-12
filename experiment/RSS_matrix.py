import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# --- 設定 ---
CSV_PATH = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\frame\score_results.csv"
RSS_THRESH = 2500
LABELS_2CLASS = ['straight', 'turn']

# --- データ読み込み ---
df = pd.read_csv(CSV_PATH)
y_true = df['gt_label'].values

# --- RSS 予測ラベル生成 ---
y_pred_rss = []
for rss in df['rss']:
    if not np.isfinite(rss):
        y_pred_rss.append('no_detection')
    else:
        y_pred_rss.append('straight' if rss <= RSS_THRESH else 'turn')
y_pred_rss = np.array(y_pred_rss)

# --- straight/turn のみ抽出 ---
mask = (y_true != 'no_detection') & (y_pred_rss != 'no_detection')
y_true_2 = y_true[mask]
y_pred_2 = y_pred_rss[mask]

# --- 混同行列（行正規化付き）計算 ---
cm_counts = confusion_matrix(y_true_2, y_pred_2, labels=LABELS_2CLASS)
cm = cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True)

# --- Matplotlib で可視化 ---
fig, ax = plt.subplots(figsize=(5,4))
im = ax.imshow(cm, cmap=plt.cm.Blues)
fig.colorbar(im, ax=ax)

# 軸ラベル・目盛りフォントサイズ
ax.set_xticks(np.arange(len(LABELS_2CLASS)))
ax.set_xticklabels(LABELS_2CLASS, fontsize=12)
ax.set_yticks(np.arange(len(LABELS_2CLASS)))
ax.set_yticklabels(LABELS_2CLASS, fontsize=12)
ax.set_xlabel('Predicted', fontsize=14)
ax.set_ylabel('True', fontsize=14)

# セル内に小数表示
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i,
            f"{cm[i, j]:.2f}",
            ha='center', va='center',
            fontsize=12,
            color='white' if cm[i, j] > thresh else 'black'
        )

plt.tight_layout()
plt.show()

# --- 2クラスの詳細レポート ---
print(classification_report(
    y_true_2, y_pred_2,
    target_names=LABELS_2CLASS,
    digits=4
))

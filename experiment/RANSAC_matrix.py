import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

# --- 設定 ---
CSV_PATH = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\frame\score_results.csv"
RANSAC_THRESH = 0.53
LABELS_2CLASS = ['straight', 'turn']

# --- データ読み込み ---
df     = pd.read_csv(CSV_PATH)
y_true = df['gt_label'].values

# --- RANSAC 予測ラベル生成 ---
y_pred = []
for ir in df['inlier_ratio']:
    if ir == 0.0:
        y_pred.append('no_detection')
    else:
        y_pred.append('straight' if ir >= RANSAC_THRESH else 'turn')
y_pred = np.array(y_pred)

# --- straight/turn のみを抽出 ---
mask = (y_true != 'no_detection') & (y_pred != 'no_detection')
y_true_2 = y_true[mask]
y_pred_2 = y_pred[mask]

# --- 混同行列の計算（2クラス版、割合に正規化）---
cm_counts = confusion_matrix(y_true_2, y_pred_2, labels=LABELS_2CLASS)
# 各行（実際のラベル）で正規化
cm = cm_counts.astype(float) / cm_counts.sum(axis=1, keepdims=True)

# --- Matplotlib でプロット ---
fig, ax = plt.subplots(figsize=(10,8))
im = ax.imshow(cm, cmap=plt.cm.Blues)
cbar = fig.colorbar(im, ax=ax)
cbar.ax.tick_params(labelsize=40)
#plt.colorbar().set_label(fontsize = 30)

# 軸ラベル・目盛り文字サイズ
ax.set_xticks(np.arange(len(LABELS_2CLASS)))
ax.set_xticklabels(LABELS_2CLASS, fontsize=40)
ax.set_yticks(np.arange(len(LABELS_2CLASS)))
ax.set_yticklabels(LABELS_2CLASS, fontsize=40)
ax.set_xlabel('Predicted', fontsize=45)
ax.set_ylabel('True', fontsize=45)

# セル内に小数表記で割合を表示
thresh = cm.max() / 2
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        ax.text(
            j, i,
            f"{cm[i, j]:.2f}",      # 小数表記（例：0.32）
            ha='center', va='center',
            fontsize=60,            # セル内文字サイズ
            color='white' if cm[i, j] > thresh else 'black'
        )

plt.tight_layout()
plt.show()

import os
import cv2
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor, LinearRegression
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# --- 設定 ---
IMG_DIR         = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\distribute\frame\raw_only"
MODEL_PATH      = r"C:\Users\ata3357\Desktop\zemi_win\block_result\rans22\weights\best.pt"
CONF_THRESH     = 0.2
FOV_DEG         = 360
RANSAC_THRESH_PX= 5.0
INLIER_THRESH   = 0.55
ROI_ANGLES      = [20, 30, 40, 50, 60, 75, 90]
SAVE_DIR        = os.path.join(IMG_DIR, "roi_search_result")
os.makedirs(SAVE_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)
results = []

for roi_angle in ROI_ANGLES:
    y_true, y_pred = [], []
    for fname in sorted(os.listdir(IMG_DIR)):
        if not fname.endswith("_raw.png"):
            continue
        gt_label = fname.split("_")[2]
        img_path = os.path.join(IMG_DIR, fname)
        img = cv2.imread(img_path)
        h, w = img.shape[:2]

        res = model.predict(img, conf=CONF_THRESH, imgsz=(w,h), verbose=False)[0]
        boxes = res.boxes.xyxy.cpu().numpy()
        centers = []
        for x1,y1,x2,y2 in boxes:
            cx = (x1+x2)/2
            theta = (cx - w/2) / w * FOV_DEG
            if abs(theta) <= roi_angle:
                cy = (y1+y2)/2
                centers.append((cx, cy))

        if len(centers) >= 2:
            data = np.array(centers)
            X = data[:,0].reshape(-1,1)
            Y = data[:,1]
            ransac = RANSACRegressor(LinearRegression(), residual_threshold=RANSAC_THRESH_PX, random_state=0)
            ransac.fit(X, Y)
            inlier_ratio = ransac.inlier_mask_.sum() / len(data)
        else:
            inlier_ratio = 0.0

        pred_label = "straight" if inlier_ratio >= INLIER_THRESH else "turn"
        y_true.append(gt_label)
        y_pred.append(pred_label)

    cm = confusion_matrix(y_true, y_pred, labels=["straight", "turn"])
    acc = (cm[0,0] + cm[1,1]) / cm.sum()
    results.append((roi_angle, acc, cm))

# --- 最良ROI角度の混同行列可視化 ---
df_result = pd.DataFrame(results, columns=["ROI_Angle", "Accuracy", "ConfusionMatrix"])
best_idx = df_result["Accuracy"].idxmax()
best_angle = df_result.iloc[best_idx]["ROI_Angle"]
best_cm    = df_result.iloc[best_idx]["ConfusionMatrix"]
best_acc   = df_result.iloc[best_idx]["Accuracy"]

disp = ConfusionMatrixDisplay(confusion_matrix=best_cm, display_labels=["straight", "turn"])
disp.plot(cmap='Blues', values_format='d')
plt.title(f"Best ROI ±{best_angle}° (Accuracy={best_acc:.3f})")
plt.savefig(os.path.join(SAVE_DIR, f"confusion_matrix_roi{int(best_angle)}.png"))

# --- Accuracy 曲線保存 ---
plt.figure()
plt.plot(df_result["ROI_Angle"], df_result["Accuracy"], marker='o')
plt.xlabel("ROI Angle (±deg)")
plt.ylabel("Accuracy")
plt.title("Accuracy vs ROI Angle")
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, "roi_accuracy_plot.png"))

# --- CSV保存 ---
csv_out = os.path.join(SAVE_DIR, "roi_search_results.csv")
df_result.drop(columns=["ConfusionMatrix"]).to_csv(csv_out, index=False)
print(f"✅ ROI最適化完了！結果: {csv_out}")

import pandas as pd

csv_path = r"C:\Users\ata3357\Desktop\zemi_win\CR-chair\pywhill\my_research\logs\kakukaku\log_mavg_20250708_125229.csv"
df = pd.read_csv(csv_path)

total_time = df['frame_time'].sum()
print(f"Total run time: {total_time:.4f}s")

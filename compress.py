import pandas as pd

# Load your original large CSV
df = pd.read_csv("data/processed/ball_by_ball_processed.csv")

# Save directly as a compressed zip (or use 'my_dataset.csv.gz' for gzip)
df.to_csv("data/processed/ball_by_ball_processed.zip", index=False, compression="zip")

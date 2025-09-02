import pandas as pd
from sklearn.preprocessing import LabelEncoder
import pickle
import os

# Load your cleaned dataset
csv_path = r"E:\Python Projects\startech product training\product_classifier\backend\startech_product_cleaned_dataset.csv"

df = pd.read_csv(csv_path)

# Fit LabelEncoder on category column
le = LabelEncoder()
le.fit(df['category'])

# Save to extracted_model folder
save_path = os.path.join("model", "extracted_model", "label_encoder.pkl")
with open(save_path, "wb") as f:
    pickle.dump(le, f)

print(f"Label encoder saved successfully with {len(le.classes_)} classes at: {save_path}")

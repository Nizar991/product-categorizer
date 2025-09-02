import zipfile
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle
import re
import pandas as pd  # only for pd.isna check

# ---------- Cleaning Functions ----------
def clean_text(s, punct_keep='.-'):   # ← no backslash
    if pd.isna(s): return ''
    s = s.lower().strip()
    pattern = rf'[^a-z0-9\s{re.escape(punct_keep)}]'
    s = re.sub(pattern, ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def clean_row(brand, title, desc, short_desc, spec):
    return ' '.join([
        clean_text(brand, ''),
        clean_text(title, '.'),
        clean_text(desc, '.-'),
        clean_text(short_desc, '.-'),
        clean_text(spec, '.-'),
    ]).strip()

# ---------- Paths ----------
model_folder = "model"
zip_path = os.path.join(model_folder, "quantized_model.zip")
extract_dir = os.path.join(model_folder, "extracted_model")

# Step 1: Extract zip
if not os.path.exists(extract_dir):
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_dir)
    print(f"Model extracted to {extract_dir}")
else:
    print(f"Model already extracted at {extract_dir}")

# Step 2: Load tokenizer
tokenizer = DistilBertTokenizer.from_pretrained(extract_dir)

# Step 3: Load label encoder
label_encoder_path = os.path.join(extract_dir, "label_encoder.pkl")
with open(label_encoder_path, "rb") as f:
    label_encoder = pickle.load(f)

num_labels = len(label_encoder.classes_)
print(f"Number of labels: {num_labels}")

# Step 4: Load model (handle both state_dict and full-model checkpoint)
model_path = os.path.join(extract_dir, "quantized_model.pth")
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

model = None

if isinstance(checkpoint, dict):
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model = DistilBertForSequenceClassification.from_pretrained(extract_dir, num_labels=num_labels)
    model.load_state_dict(state_dict)
elif isinstance(checkpoint, torch.nn.Module):
    print("Checkpoint is a full model object. Using it directly.")
    model = checkpoint
else:
    raise RuntimeError(f"Unrecognized checkpoint type: {type(checkpoint)}")

# Ensure model is on CPU and in eval mode
model.to(torch.device("cpu"))
if not hasattr(model.config, "output_attentions"):
    model.config.output_attentions = getattr(model.config, "_output_attentions", False)
if not hasattr(model.config, "output_hidden_states"):
    model.config.output_hidden_states = getattr(model.config, "_output_hidden_states", False)
if not hasattr(model.config, "return_dict"):
    model.config.return_dict = getattr(model.config, "use_return_dict", True)
model.eval()

print("Model, tokenizer, and label encoder loaded successfully!")

# ---------- User Input Simulation ----------
print("\nEnter product details below (press Enter to skip a field):")
brand      = input("Brand: ")
title      = input("Title: ")
description= input("Description: ")
short_desc = input("Short description: ")
spec       = input("Specification: ")

# Clean & merge (like training)
combined_text = clean_row(brand, title, description, short_desc, spec)
if combined_text == "":
    print("No input provided. Exiting.")
    exit(0)

print(f"\nCombined & Cleaned Text (first 300 chars):\n{combined_text[:300]}...\n")

# ---------- Prediction ----------
inputs = tokenizer(
    combined_text,
    return_tensors="pt",
    truncation=True,
    padding=True,
    max_length=512
)

with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits
    predicted_class_id = torch.argmax(logits, dim=1).item()

predicted_category = label_encoder.inverse_transform([predicted_class_id])[0]
print(f"✅ Predicted Category: {predicted_category}")

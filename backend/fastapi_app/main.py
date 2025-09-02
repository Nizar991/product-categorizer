# E:\Python Projects\startech product training\product_classifier\backend\fastapi_app\main.py
import zipfile
import os
import torch
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
import pickle
import re
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
import requests

# ---------- FastAPI app ----------
app = FastAPI(title="Product Category Predictor")

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"]
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Input schema ----------
class ProductInput(BaseModel):
    brand: str = ""
    title: str = ""
    description: str = ""
    short_desc: str = ""
    spec: str = ""

# ---------- Cleaning functions ----------
def clean_text(s, punct_keep='.-'):
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
model_folder = os.path.join(os.path.dirname(__file__), "..", "model")
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

# Step 4: Load model
model_path = os.path.join(extract_dir, "quantized_model.pth")
checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)

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
    model = checkpoint
else:
    raise RuntimeError(f"Unrecognized checkpoint type: {type(checkpoint)}")

# Ensure CPU & eval
model.to(torch.device("cpu"))
if not hasattr(model.config, "output_attentions"):
    model.config.output_attentions = getattr(model.config, "_output_attentions", False)
if not hasattr(model.config, "output_hidden_states"):
    model.config.output_hidden_states = getattr(model.config, "_output_hidden_states", False)
if not hasattr(model.config, "return_dict"):
    model.config.return_dict = getattr(model.config, "use_return_dict", True)
model.eval()
print("Model, tokenizer, and label encoder loaded successfully!")

# ---------- Strapi config ----------
# STRAPI_URL = "http://localhost:1337/api/products"
STRAPI_URL = "http://strapi:1337/api/products"
STRAPI_API_TOKEN = "ENTER THE API URL" #put your api token url here

# ---------- FastAPI endpoint ----------
@app.post("/predict_category")
def predict_category(input: ProductInput):
    combined_text = clean_row(
        input.brand, input.title, input.description, input.short_desc, input.spec
    )
    if combined_text == "":
        return {"error": "No input provided"}

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

    # ---------- Send to Strapi ----------
    payload = {
        "data": {
            "brand": input.brand,
            "title": input.title,
            "description": input.description,
            "short_description": input.short_desc,
            "specification": input.spec,
            "combined_text": combined_text,
            "category": predicted_category
        }
    }
    headers = {
        "Authorization": f"Bearer {STRAPI_API_TOKEN}",
        "Content-Type": "application/json"
    }

    print("Payload to Strapi:", payload)

    try:
        response = requests.post(STRAPI_URL, json=payload, headers=headers)
        print("Strapi response:", response.status_code, response.text)
        if response.status_code in [200, 201]:
            # Call n8n webhook
            n8n_url = "http://n8n_app:5678/webhook/supabase-update"
            n8n_payload = {
                'table': 'products',
                'action': 'INSERT', 
                'data': payload["data"]
            }
            try:
                n8n_response = requests.post(n8n_url, json=n8n_payload)
                print("n8n response:", n8n_response.status_code, n8n_response.text)
            except Exception as e:
                print(f"n8n error: {str(e)}")
            strapi_status = "Saved to Strapi successfully"
        else:
            strapi_status = f"Failed to save to Strapi: {response.text}"
    except Exception as e:
        strapi_status = f"Error connecting to Strapi: {str(e)}"

    return {
        "predicted_category": predicted_category,
        "strapi_status": strapi_status
    }

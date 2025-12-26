from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image
from rapidfuzz import process, fuzz
import pandas as pd
import numpy as np
import cv2
import io
import re
import os

app = FastAPI(title="MedScript HTR Final API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- 1. SETUP & DATABASE LOADING ---
print("Initializing System...")

DRUG_LIST = []

try:
    if os.path.exists('drug_database.csv'):
        # Load the large database
        print("Loading drug database... this might take a moment.")
        drug_df = pd.read_csv('drug_database.csv')
        
        # Ensure data is string type and remove empty values
        brands = drug_df['PROPRIETARYNAME'].dropna().astype(str).tolist()
        generics = drug_df['NONPROPRIETARYNAME'].dropna().astype(str).tolist()
        
        # Combine and remove duplicates for the search pool
        DRUG_LIST = list(set(brands + generics))
        print(f"Database Loaded Successfully: {len(DRUG_LIST)} unique entries.")
    else:
        print("Warning: 'drug_database.csv' not found. System will rely on manual list.")
        DRUG_LIST = ["Microdon DT", "Espra", "Deriva MS", "Panadol", "Amoxicillin"] # Fallback
except Exception as e:
    print(f"Database Error: {e}")
    DRUG_LIST = []

# Load AI Model
try:
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    print("AI Model Ready!")
except Exception as e:
    print(f"Model Error: {e}")

# --- 2. CORE LOGIC ---

def clean_and_match(raw_text):
    """
    Cleans OCR text and applies fuzzy matching against the loaded drug list.
    """
    # Basic cleanup
    text = raw_text.replace('</s>', '').replace('s>', '')
    text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
    
    # Manual Correction Dictionary (for very hard to read handwriting)
    corrections = {
        "macradon": "Microdon DT",
        "mecrodon": "Microdon DT",
        "macnahon": "Microdon DT",
        "megadote": "mg"
    }
    
    lower_raw = text.lower()
    for wrong, right in corrections.items():
        if wrong in lower_raw:
            return right, [{"name": right, "score": 95.0}]

    # Filter noise words
    noise_words = [
        'mg', 'mgs', 'ml', 'tablet', 'capsule', 'tabs', 
        'nocte', 'daily', 'bd', 'tds', 'dr',
        'waste', 'face', 'taste', 'get', 'of', 'negricte'
    ]
    
    words = text.split()
    filtered = [
        w for w in words 
        if (len(w) > 2 and w.lower() not in noise_words) 
        or w.upper() in ['DT', 'SR', 'XR', 'MS', 'DS', 'LA']
    ]
    
    clean_query = " ".join(filtered).strip()
    
    if not clean_query:
        return None, []

    # Fuzzy Matching against the 24k+ list
    # limit=5 gets top 5 matches
    matches = process.extract(clean_query, DRUG_LIST, scorer=fuzz.token_set_ratio, limit=5)
    suggestions = [{"name": m[0], "score": round(m[1], 2)} for m in matches]
    
    return clean_query, suggestions

def segment_lines_from_bytes(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Dilation kernel to connect letters
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (80, 1))
    dilated = cv2.dilate(thresh, kernel, iterations=1)
    
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bounding_boxes = [cv2.boundingRect(c) for c in contours]
    (contours, bounding_boxes) = zip(*sorted(zip(contours, bounding_boxes), key=lambda b: b[1][1]))
    
    crops = []
    for box in bounding_boxes:
        x, y, w, h = box
        if w > 30 and h > 10:
            crop = image[max(0, y-5):min(image.shape[0], y+h+5), max(0, x-5):min(image.shape[1], x+w+5)]
            crops.append(Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)))
            
    return crops

@app.post("/predict-full")
async def predict_full(file: UploadFile = File(...)):
    contents = await file.read()
    line_images = segment_lines_from_bytes(contents)
    results = []
    
    for i, img in enumerate(line_images):
        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        raw_text = processor.batch_decode(generated_ids, skip_special_characters=True)[0]
        
        cleaned, suggestions = clean_and_match(raw_text)
        
        if cleaned:
            results.append({
                "id": i,
                "raw": raw_text,
                "cleaned": cleaned,
                "suggestions": suggestions
            })
            
    return {"lines_detected": len(results), "data": results}
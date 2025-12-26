from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image, ImageOps
import torch
import io
import pandas as pd
from rapidfuzz import process, fuzz
import cv2
import numpy as np
import re

app = FastAPI()

# --- CORS Settings ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading Model... Please wait.")
processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
print("Model Loaded Successfully!")

# --- Load Database ---
try:
    df = pd.read_csv("drug_database.csv")
    drug_list = df.iloc[:, 0].astype(str).tolist()
    print(f"Loaded {len(drug_list)} drugs from database.")
except Exception as e:
    print(f"Error loading database: {e}")
    drug_list = []

# --- IGNORED WORDS LIST (Expanded for common OCR errors) ---
IGNORED_WORDS = [
    "nocte", "nocto", "bd", "bid", "tds", "tid", "od", "sos", "stat", 
    "mg", "ml", "g", "kg", "mcg", "tab", "cap", "capsule", "tablet",
    "face", "gel", "cream", "oint", "ointment", "drops", "syrup",
    "before", "after", "food", "meals", "daily", "days", "months",
    "date", "age", "sex", "name", "dr", "rx", "signature"
]

# --- SEGMENTATION LOGIC ---
def segment_lines(image_bytes):
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    
    # Kernel (40, 2) ensures lines don't merge vertically
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 2)) 
    dilate = cv2.dilate(thresh, kernel, iterations=1)
    
    cnts = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # Sort contours top to bottom
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    if len(cnts) > 0:
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes), key=lambda b:b[1][1]))

    line_images = []
    
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        # Filter small noise
        if h > 20 and w > 50:
            roi = img[y:y+h, x:x+w] 
            pil_img = Image.fromarray(cv2.cvtColor(roi, cv2.COLOR_BGR2RGB))
            # Add padding for better OCR accuracy
            padded_img = ImageOps.expand(pil_img, border=20, fill='white')
            line_images.append(padded_img)

    return line_images

# --- NEW: ADVANCED MATCHING FUNCTION ---
def find_best_drug_match(text, drug_db):
    # 1. Cleaning: Lowercase and remove numbers (prevents matching "100" to "GABA 100")
    # We keep only letters for the matching logic
    clean_text = text.lower()
    
    # Remove ignored words
    for bad_word in IGNORED_WORDS:
        clean_text = clean_text.replace(f" {bad_word} ", " ")
        if clean_text.endswith(f" {bad_word}"): clean_text = clean_text[:-len(bad_word)-1]
        if clean_text.startswith(f"{bad_word} "): clean_text = clean_text[len(bad_word)+1:]
    
    # Remove digits completely for matching
    text_alpha_only = re.sub(r'[^a-zA-Z\s]', '', clean_text).strip()
    
    if len(text_alpha_only) < 3:
        return None, 0, "Ignored (Noise)"

    # 2. Get Top 5 Candidates (Not just one)
    # scorer=fuzz.WRatio handles partial matches well
    matches = process.extract(text_alpha_only, drug_db, scorer=fuzz.WRatio, limit=5)

    best_candidate = None
    best_score = 0
    status = "Not Found"

    # 3. Iterate through candidates to find the first valid one
    for name, score, idx in matches:
        # Rule 1: First Letter Match
        # If OCR is "Macrodon" (M) and DB is "Microdon" (M) -> Pass
        # If OCR is "Macrodon" (M) and DB is "GABA" (G) -> Fail
        
        db_first_char = name.lower()[0]
        ocr_first_char = text_alpha_only[0]

        if ocr_first_char == db_first_char:
            # If letters match, we accept a lower score (e.g., 50%)
            if score >= 50:
                best_candidate = name
                best_score = score
                status = "Found"
                break # Found the best valid match, stop looking
        else:
            # If letters don't match, we need a very high score (e.g., 85%) to accept
            # This handles cases where the first letter is read wrong (e.g., "P" vs "F")
            if score >= 85:
                best_candidate = name
                best_score = score
                status = "Found (High Conf)"
                break
    
    return best_candidate, best_score, status

@app.post("/predict-full")
async def predict(file: UploadFile = File(...)):
    image_data = await file.read()
    lines = segment_lines(image_data)
    
    if not lines:
        raw_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        lines = [ImageOps.expand(raw_img, border=20, fill='white')]

    results = []
    print(f"Processing {len(lines)} lines...")

    for i, line_img in enumerate(lines):
        # OCR Prediction
        pixel_values = processor(images=line_img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        raw_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        # Use the new Smart Matching Logic
        drug_name, confidence, status = find_best_drug_match(raw_text, drug_list)

        if status == "Ignored (Noise)":
            print(f"Line {i+1} IGNORED: {raw_text}")
            continue
            
        if status == "Not Found":
             print(f"Line {i+1} NO MATCH: {raw_text}")
        else:
             print(f"Line {i+1}: {raw_text} -> {drug_name} ({confidence}%)")
             
             # Add to results
             results.append({
                "raw_text": raw_text,
                "drug_name": drug_name,
                "confidence": confidence,
                "status": status
             })

    return results
# MedScript HTR - AI Prescription Digitizer 🏥💊

MedScript HTR is an AI-powered system designed to digitize handwritten medical prescriptions. It uses **Local Deep Learning (PaddleOCR)** to recognize text and a custom **Fuzzy Logic Algorithm** to verify drug names against a pharmacy database. 

**This project runs 100% locally and does not require an internet connection for AI processing.**

---

## 🚀 Key Features

* **Offline Handwriting Recognition:** Uses `PaddleOCR` (PP-OCRv4) to detect and recognize cursive handwriting locally.
* **Smart Drug Matching:** A fuzzy string matching algorithm (`RapidFuzz`) corrects OCR errors by cross-referencing with a verified drug database.
* **Advanced Preprocessing:** Custom OpenCV logic for noise removal, binarization, and letter connection (to handle cursive text).
* **Privacy Focused:** No data is sent to the cloud. Everything is processed on your machine.

---

## 🛠 Tech Stack

* **Frontend:** React.js, Tailwind CSS
* **Backend:** Python (FastAPI)
* **AI/ML Engine:** PaddleOCR (Deep Learning)
* **Image Processing:** OpenCV, NumPy
* **Data Matching:** RapidFuzz (Levenshtein Distance Logic)

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8 or higher
* Node.js & npm

### 1. Backend Setup (Python)

Navigate to the backend folder and set up the virtual environment.

```bash
cd backend
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
.\venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
# (Make sure to install paddlepaddle and paddleocr if not in requirements)
pip install paddlepaddle paddleocr fastapi uvicorn python-multipart rapidfuzz pandas opencv-python-headless

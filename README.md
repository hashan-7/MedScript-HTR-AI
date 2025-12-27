# MedScript HTR - AI Prescription Digitizer 🏥💊

MedScript HTR is an AI-powered system designed to digitize handwritten medical prescriptions. It uses **Local Deep Learning (PaddleOCR)** to recognize text and a custom **Fuzzy Logic Algorithm** to verify drug names against a pharmacy database.

**This project runs 100% locally and does not require an internet connection for AI processing.**

---

## 🚀 Key Features

* **Offline Handwriting Recognition:** Uses `PaddleOCR` (PP-OCRv5) to detect and recognize cursive handwriting locally without external APIs.

* **Smart Drug Matching:** A fuzzy string matching algorithm (`RapidFuzz`) corrects OCR errors by cross-referencing extracted text with a verified local drug database.

* **Advanced Preprocessing:** Custom OpenCV logic including adaptive thresholding and morphological dilation to handle complex, cursive handwriting.

* **Privacy Focused:** No patient data is sent to the cloud. Everything is processed securely on your local machine.

---

## 🛠 Tech Stack

* **Frontend:** React.js, Tailwind CSS
* **Backend:** Python (FastAPI)
* **AI/ML Engine:** PaddleOCR (Deep Learning)
* **Image Processing:** OpenCV, NumPy
* **Data Matching:** RapidFuzz (Levenshtein Distance Logic)

---

## ⚙️ Installation & Setup

Follow these steps to set up the project locally.

### Prerequisites

* Python 3.8 or higher
* Node.js & npm
* Git

### 1. Backend Setup (Python)

Navigate to the backend folder and set up the virtual environment.

```
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# For Windows:
.\venv\Scripts\activate
# For Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install paddlepaddle paddleocr fastapi uvicorn python-multipart rapidfuzz pandas opencv-python-headless

# Run the Backend Server:

uvicorn main:app --reload --port 8000
The API will start at http://127.0.0.1:8000
```
```
# Frontend Setup (React)
Open a new terminal and navigate to the frontend folder.

cd frontend

# Install dependencies
npm install

# Start the development server
npm start
The application will open automatically at http://localhost:3000
```

### 📸 How to Use

 Launch the App: Ensure both Backend and Frontend terminals are running.

 * Upload Image: Upload a clear image of a handwritten prescription (JPG/PNG).

 * Processing: The system uses OpenCV to preprocess the image (thickening ink strokes) and PaddleOCR to read the text.

 * Verification: The extracted text is automatically compared against drug_database.csv.

 * Found: High confidence match.

 * Smart Match: Medium confidence but medically valid match.

 * Not Found: The text did not match any known drug.

 * Result: View the digitized list and export it if needed.


### 🧠 How It Works (Under the Hood)

* Morphological Dilation: Since cursive handwriting often has connected letters that OCR models miss, we use OpenCV to slightly "thicken" and connect the letters before processing.

* PaddleOCR Engine: We use the PP-OCRv5 model which is optimized for document scene text recognition and uses a CRNN architecture.

* Fuzzy Logic Layer: Raw OCR text is rarely perfect. We use RapidFuzz to calculate the similarity between the OCR output and the drug database.

* Example: If OCR reads "Renyer" as "Renger", the fuzzy logic identifies "Renyer" as the correct drug based on the string distance and database presence.

Made by H7 
📝 License : 
This project is developed for educational purposes.



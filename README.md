# 🏥 MedScript HTR - AI Prescription Digitizer

MedScript HTR is an AI-powered system designed to digitize handwritten medical prescriptions. It uses **Microsoft TrOCR** for handwriting recognition and a custom **Fuzzy Logic algorithm** to match detected text against a massive drug database, ensuring high accuracy even with messy handwriting.

![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
![React](https://img.shields.io/badge/React-18-blue?logo=react)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED?logo=docker)
![FastAPI](https://img.shields.io/badge/FastAPI-Backend-009688?logo=fastapi)

## 🚀 Key Features
* **AI Handwriting Recognition:** Uses TrOCR (Transformer-based Optical Character Recognition).
* **Drug Verification:** Matches OCR output with a localized database of 24,000+ medicines using Fuzzy Matching (RapidFuzz).
* **Real-time Camera Integration:** Capture prescriptions directly via webcam/mobile.
* **PDF Reporting:** Generates professional PDF reports of the digitized prescription.
* **Dockerized:** Fully containerized for easy deployment.

## 🛠️ Tech Stack
* **Frontend:** React.js, Bootstrap
* **Backend:** Python FastAPI, PyTorch, Uvicorn
* **AI Model:** Microsoft TrOCR (Fine-tuned on IAM dataset)
* **Data Processing:** Pandas, RapidFuzz, OpenCV

## 📦 How to Run (Using Docker)

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourName/MedScript-HTR-AI.git](https://github.com/YourName/MedScript-HTR-AI.git)
    cd MedScript-HTR-AI
    ```

2.  **Run with Docker Compose:**
    ```bash
    docker-compose up --build
    ```

3.  **Access the App:**
    * Frontend: `http://localhost:3000`
    * Backend API: `http://localhost:8000/docs`

## 👨‍💻 Developer
Developed by **[Your Name]** as a Full Stack AI Project.
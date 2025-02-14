# MANTHAN: AI Medical Assistant

## Overview

MANTHAN is an AI-powered medical assistant designed to assist with healthcare diagnostics, report summarization, and intelligent Q&A. This application integrates deep learning and natural language processing (NLP) to enhance efficiency in healthcare and knowledge retrieval.

## Features

- 📊 **Summarize Medical PDFs & YouTube Lectures**
- 🩻 **Detect Breast Cancer from Mammograms**
- 🦠 **Diagnose Malaria from Blood Smears**
- 💬 **AI Chatbot for Medical Q&A**

## Technologies Used

- **Programming Language:** Python
- **Framework:** Streamlit
- **Machine Learning:** TensorFlow, Keras
- **Natural Language Processing:** Ollama, Whisper, Facebook BART
- **YouTube Video Summarization:** yt_dlp, youtube_transcript_api
- **PDF Processing:** pdfplumber, pymupdf

## Installation

### Prerequisites

Ensure you have Python installed on your system.

### Steps to Install

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/manthan.git
   cd manthan
   ```
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the application:
   ```bash
   streamlit run app.py
   ```

## Usage

- Upload a **PDF** document for medical report summarization.
- Provide a **YouTube video link** to extract key insights from medical lectures.
- Upload **Mammogram images** for breast cancer detection.
- Upload **Blood smear images** for malaria detection.
- Use the **AI-powered chatbot** to get medical assistance.

## Project Structure

```
.
├── app.py                 # Main application file
├── requirements.txt       # Required dependencies
├── modules/               # Contains different AI modules
│   ├── breast_cancer_detection.py
│   ├── Malaria_detection.py
│   ├── pdf_summary.py
│   ├── youtube_summary2.py
└── README.md              # Project documentation
```

## Explanation of Modules and Models

### 1. Breast Cancer Detection (`breast_cancer_detection.py`)
- **Purpose:** Detect breast cancer from mammogram images using deep learning.
- **Model Used:** Convolutional Neural Network (CNN).
- **Key Steps:**
  1. **Preprocessing:** Normalizes images, applies augmentation (rotation, flipping, etc.).
  2. **Feature Extraction:** Uses CNN layers to detect patterns in histopathological images.
  3. **Classification:** Uses Softmax or Sigmoid activation to classify images as benign or malignant.
  4. **Evaluation:** Uses metrics like accuracy, precision, and recall.

### 2. Malaria Detection (`Malaria_detection.py`)
- **Purpose:** Identify malaria-infected blood smear images.
- **Model Used:** CNN-based classifier.
- **Key Steps:**
  1. **Data Preprocessing:** Converts images to grayscale, enhances contrast.
  2. **Feature Extraction:** Detects patterns and parasite structures using CNN layers.
  3. **Prediction:** Classifies blood smear images as infected or non-infected.
  4. **Model Optimization:** Uses dropout layers to reduce overfitting.

### 3. PDF Summarization (`pdf_summary.py`)
- **Purpose:** Extracts and summarizes key insights from medical PDFs.
- **Model Used:** Transformer-based summarization (Ollama).
- **Key Steps:**
  1. **Text Extraction:** Uses `pdfplumber` to extract text.
  2. **Preprocessing:** Cleans extracted text (removes noise, tokenization).
  3. **Summarization:** Applies NLP techniques to generate concise summaries.
  4. **Output Generation:** Displays key points from medical reports.

### 4. YouTube Lecture Summarization (`youtube_summary2.py`)
- **Purpose:** Summarizes YouTube videos related to medical topics.
- **Model Used:** Facebook BART.
- **Key Steps:**
  1. **Transcript Extraction:** Uses `youtube_transcript_api` to fetch video subtitles.
  2. **Text Processing:** Cleans and structures transcript data.
  3. **Summarization:** Generates an abstract summary of the content.
  4. **Display Output:** Presents a structured summary for quick review.

### 5. AI-Powered Chatbot
- **Purpose:** Answers medical queries in real-time.
- **Model Used:** Llama 3.2 (or similar large language models).
- **Key Steps:**
  1. **User Input Processing:** Detects the type of query.
  2. **Medical Query Restriction:** Ensures only medical-related questions are answered.
  3. **Response Generation:** Uses a pre-trained transformer model for medical Q&A.
  4. **Chat History Management:** Maintains a session-based conversation history.

## Contributors

- **Harsh Jain**
- **Dhananjay Gaikwad**
- **Abhishek Mali**
- **Siddhi Shintre**

## License

This project is licensed under the MIT License.

## Acknowledgment

This project was developed as part of the **PG-Diploma in Artificial Intelligence** at **C-DAC ACTS (Pune)** under the guidance of **Mr. Prakash Sinha**.


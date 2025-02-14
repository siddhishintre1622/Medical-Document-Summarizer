# import pdfplumber
# import streamlit as st
# from ollama import Client

# client = Client()

# def extract_text_from_pdf(pdf_file):
#     text = ""
#     try:
#         with pdfplumber.open(pdf_file) as pdf:
#             text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
#     except Exception as e:
#         st.error(f"Error reading PDF: {e}")
#     return text

# def summarize_medical_report(text):
#     if text:
#         response = client.chat(
#             model="llama3.2:latest",
#             messages=[{"role": "user", "content": f"Summarize this medical report in simple words:\n{text}"}]
#         )
#         return response['message']['content']
#     else:
#         return "Could not extract text from the PDF. It may be an image-based PDF."

# def process_pdf_upload(pdf_file):
#     if pdf_file:
#         extracted_text = extract_text_from_pdf(pdf_file)
#         summary = summarize_medical_report(extracted_text)
#         return summary
#     return None

import pdfplumber
import streamlit as st
from ollama import Client

client = Client()

def extract_text_from_pdf(pdf_file):
    text = ""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
    return text

def summarize_medical_report(text):
    if text:
        response = client.chat(
            model="llama3.2:latest",
            messages=[{"role": "user", "content": f"Summarize this medical report in simple words and return the key points in bullet format:\n{text}"}]
        )
        # Ensure the response is in bullet points
        summary = response['message']['content']
        bullet_summary = "\n".join([f"- {line}" for line in summary.split("\n") if line.strip()])
        return bullet_summary
    else:
        return "Could not extract text from the PDF. It may be an image-based PDF."

def process_pdf_upload(pdf_file):
    if pdf_file:
        extracted_text = extract_text_from_pdf(pdf_file)
        summary = summarize_medical_report(extracted_text)
        return summary
    return None

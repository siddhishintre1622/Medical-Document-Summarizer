import streamlit as st
import mimetypes
from ollama import Client
from modules.breast_cancer_detection import load_trained_model, predict_cancer
from modules.pdf_summary import process_pdf_upload
from modules.youtube_summary2 import youtube_summary_app
from modules.Malaria_detection import preprocess_image, make_prediction
from tensorflow.keras.models import load_model
import tempfile

# Initialize Llama 3.2 chatbot
client = Client()

# Set Streamlit page config
st.set_page_config(page_title="MANTHAN: AI Medical Assistant", page_icon="🩺", layout="wide")
st.title("🩺MANTHAN: AI Medical Diagnosis Assistant Chatbot")

# Sidebar introduction
st.sidebar.title("🩺 AI Medical Diagnosis Assistant Chatbot")
st.sidebar.header("About MedAI")
st.sidebar.markdown("""
**Welcome to MedAI!** 🤖🩺

Capabilities:
- 📊 Summarize Medical PDFs & YouTube Lectures
- 🩻 Detect Breast Cancer from Mammograms
- 🦠 Diagnose Malaria from Blood Smears
- 💬 Medical Q&A Chat Support

Your smart medical companion!
""")

# Load Malaria Model
malaria_model = load_model('C:/Users/DAI.STUDENTSDC/Downloads/Medai/modules/malaria_model.h5')

# System prompt to restrict chatbot to medical queries
system_prompt = {
    "role": "system",
    "content": "You are an AI medical assistant. Answer only medical-related queries. If someone greets like 'hi', give a friendly introduction. For non-medical questions, reply: 'I'm designed to assist only with medical-related queries.'"
}

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = [system_prompt]

# Detect file type
def detect_file_type(uploaded_file):
    if uploaded_file is None:
        return None
    if isinstance(uploaded_file, str):  # YouTube link
        return "youtube_link"
    mime_type, _ = mimetypes.guess_type(uploaded_file.name)
    if mime_type:
        if mime_type.startswith("image"):
            return "image"
        elif mime_type == "application/pdf":
            return "pdf"
    return "unknown"

# Process user request
def process_user_request(user_input):
    if "youtube.com" in user_input:
        return "youtube_link"
    elif "pdf" in user_input.lower() or "document" in user_input.lower():
        return "pdf"
    elif "detect cancer" in user_input.lower() or "mammogram" in user_input.lower():
        return "cancer_image"
    elif "detect malaria" in user_input.lower() or "blood smear" in user_input.lower():
        return "malaria_image"
    else:
        return "chat"

# Display chat history
for message in st.session_state.messages[1:]:  # Skip system prompt
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Sidebar for file upload or YouTube link
st.sidebar.header("Upload File")
file_input = st.sidebar.text_input("Enter YouTube Link (or upload a file)")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

# Detect file type
file_type = detect_file_type(file_input if file_input else uploaded_file)

# User input handling
if user_prompt := st.chat_input("How can I assist you today?"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    request_type = process_user_request(user_prompt)

    if request_type == "youtube_link":
        with st.spinner("Summarizing YouTube Lecture..."):
            summary = youtube_summary_app(file_input)
        st.session_state.messages.append({"role": "assistant", "content": summary})
        st.chat_message("assistant").markdown(summary)

    elif request_type == "pdf":
        with st.spinner("Processing PDF Summary..."):
            summary = process_pdf_upload(uploaded_file)
        st.session_state.messages.append({"role": "assistant", "content": summary})
        st.chat_message("assistant").markdown(summary)

    elif request_type == "cancer_image":
        with st.spinner("Analyzing Mammogram Image..."):
            model = load_trained_model()
            result = predict_cancer(uploaded_file, model)
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").markdown(result)

    elif request_type == "malaria_image":
        with st.spinner("Analyzing Blood Smear Image for Malaria..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
                temp_file.write(uploaded_file.read())
                temp_filename = temp_file.name
            image = preprocess_image(temp_filename)
            prediction = make_prediction(malaria_model, image)
            result = "Malaria Detected" if prediction >= 0.5 else "Malaria Not Detected"
        st.session_state.messages.append({"role": "assistant", "content": result})
        st.chat_message("assistant").markdown(result)

    else:
        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response = ""
            try:
                for response in client.chat(model="llama3.2:latest", messages=st.session_state.messages, stream=True):
                    full_response += response['message']['content']
                    message_placeholder.markdown(full_response + "▌")
                message_placeholder.markdown(full_response)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"An error occurred: {e}")

# Clear chat history button
if st.button("Clear Chat History"):
    st.session_state.messages = [system_prompt]  # Retain the system prompt

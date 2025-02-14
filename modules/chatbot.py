import streamlit as st
from ollama import Client
from breast_cancer_detection import load_trained_model, predict_cancer
from pdf_summary import process_pdf_upload
from youtube_summary import youtube_summary_app

# Initialize Ollama client
client = Client()

# Set up Streamlit page
st.set_page_config(page_title="🩺 MANTHAN : AI Medical Assistant Chatbot", page_icon="🤖", layout="wide")
st.title("🩺 MANTHAN : AI Medical Assistant Chatbot")

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Sidebar for file upload and YouTube link
st.sidebar.header("Upload File or Enter YouTube Link")
file_input = st.sidebar.text_input("Enter YouTube Link (or upload a file)")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

# Process file or YouTube link
if file_input:
    with st.spinner("Processing YouTube Link..."):
        youtube_summary_app(file_input)
elif uploaded_file:
    file_type = uploaded_file.type
    if file_type.startswith("image"):
        model = load_trained_model()
        predict_cancer(uploaded_file, model)
    elif file_type == "application/pdf":
        summary = process_pdf_upload(uploaded_file)
        st.subheader("📑 Medical Report Summary")
        st.write(summary)
    else:
        st.warning("⚠️ Please upload a valid file or enter a YouTube link.")

# Display chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for chatbot
if prompt := st.chat_input("Ask anything related to medical analysis!"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Generate AI response
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""
        for response in client.chat(model="llama3.2:latest", messages=st.session_state.messages, stream=True):
            full_response += response['message']['content']
            message_placeholder.markdown(full_response + "▌")
        message_placeholder.markdown(full_response)
    st.session_state.messages.append({"role": "assistant", "content": full_response})

# Sidebar info
st.sidebar.title("About")
st.sidebar.info("This AI Medical Assistant analyzes mammogram images, summarizes medical PDFs, and extracts insights from YouTube lectures.")
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ by Your Name")

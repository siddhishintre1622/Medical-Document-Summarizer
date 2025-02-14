import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from PIL import Image
import streamlit as st

MODEL_PATH = "C:/Users/DAI.STUDENTSDC/Downloads/Medai/modules/bc_model.h5"

def load_trained_model():
    try:
        model = load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.sidebar.error(f"Error Loading Model: {e}")
        return None

def preprocess_image(img):
    img = img.resize((50, 50))
    img_array = np.array(img) / 255.0
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def predict_cancer(uploaded_file, model):
    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)
        processed_img = preprocess_image(img)
        prediction = model.predict(processed_img)
        result_text = "Breast Cancer Detected" if prediction[0][0] > 0.5 else "No Cancer Detected"
        
        st.sidebar.subheader("Prediction Result")
        st.sidebar.markdown(f"**{result_text}**")
        st.sidebar.write(f"Confidence Score: {prediction[0][0]:.4f}")
        
        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.set_title(result_text, color="red" if prediction[0][0] > 0.5 else "green")
        ax.axis("off")
        st.pyplot(fig)

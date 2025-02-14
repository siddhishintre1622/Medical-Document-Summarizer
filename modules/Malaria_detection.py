import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import tempfile
import matplotlib.pyplot as plt
from PIL import Image

def preprocess_image(img_path, target_size=(150, 150)):
    img = load_img(img_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize
    return img_array

def make_prediction(model, image):
    prediction = model.predict(image)
    return prediction[0][0]  # Extract single value

def main():
    st.title('Malaria Detection using Deep Learning')
    st.write('Upload a blood smear image, and the model will predict whether malaria is detected or not.')
    
    model = load_model('malaria_model.h5')  # Load trained model
    
    uploaded_file = st.file_uploader('Choose an image...', type=['png', 'jpg', 'jpeg'])
    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp_file:
            temp_file.write(uploaded_file.read())
            temp_filename = temp_file.name
        
        st.image(temp_filename, caption='Uploaded Image', use_column_width=True)
        
        image = preprocess_image(temp_filename)
        prediction = make_prediction(model, image)
        img = Image.open(uploaded_file)
        fig, ax = plt.subplots()
        if prediction >= 0.5:
            ax.set_title('Malaria Detected')
            ax.imshow(image)
            st.error('Malaria Detected')
            st.pyplot(fig)
        else:
            ax.set_title('No Malaria Detected')
            ax.imshow(image)
            st.success('Malaria Not Detected')
            st.pyplot(fig)

if __name__ == '__main__':
    main()

import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Debugging: Print current working directory and files
print("Current Working Directory:", os.getcwd())
print("Files in Current Directory:", os.listdir())

# Step 1: Load the pre-trained model from the .h5 file
try:
    model = load_model('model.h5')  # Update the path if needed
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Error: model.h5 file not found. Please check the file path.")
    exit()

# Step 2: Load and prepare your test data
def preprocess_image(img_path, target_size=(150, 150)):  # Changed to match model's expected size
    img = tf.keras.preprocessing.image.load_img(img_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Example: Path to a single test image
img_path = 'C:/Users/DAI.STUDENTSDC/Downloads/Medai/C1_thinF_IMG_20150604_105100_cell_180.png'  # Replace with your image path
X_test = preprocess_image(img_path)

# Step 3: Make predictions on the test data
y_pred = model.predict(X_test)

# Step 4: Convert predictions to "Malaria Detected" or "Malaria Not Detected"
if y_pred[0][0] >= 0.5:
    print("Malaria Detected")
else:
    print("Malaria Not Detected")

import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import matplotlib.pyplot as plt
from PIL import Image

# Load the trained model (make sure the path is correct)
model = load_model(r'E:/FINAL_PROJECT/Breast_Cancer/Breast-cancer-detection-using-CNN/notebook/model.h5')

# Function to load and preprocess the image
def preprocess_image(img_path):
    # Load the image
    img = Image.open(img_path)
    
    # Resize the image to 50x50 pixels (the size expected by the model)
    img = img.resize((50, 50))
    
    # Convert the image to a numpy array and normalize
    img_array = np.array(img) / 255.0  # Normalize pixel values to [0, 1]
    
    # If the image is grayscale, convert it to RGB
    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)  # Stack grayscale to RGB
    
    # Add batch dimension (model expects batch of images, even for one)
    img_array = np.expand_dims(img_array, axis=0)
    
    return img_array

# Example image path (change this to your image path)
img_path = 'E:/FINAL_PROJECT/Breast_Cancer/Breast-cancer-detection-using-CNN/notebook/archive (13)/13591/0/13591_idx5_x101_y1_class0.png'

# Preprocess the image
processed_img = preprocess_image(img_path)

# Make a prediction
prediction = model.predict(processed_img)

# Access the prediction value and compare
# Assuming the model outputs a single value between 0 and 1
if prediction[0][0] > 0.5:
    print("The image is predicted to have Breast Cancer.")
else:
    print("The image is predicted to be benign (no Breast Cancer).")

# Optional: Display the image
img = Image.open(img_path)
plt.imshow(img)
plt.show()

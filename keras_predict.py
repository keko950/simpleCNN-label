import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image
import numpy as np
# Set parameters
batch_size = 32
img_height = 120
img_width = 60
num_epochs = 10


model = tf.keras.models.load_model('my_model.keras')

# Function to load and preprocess an image
def load_and_preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(img_height, img_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array /= 255.0  # Normalize to [0, 1]
    return img_array

# Example image path
img_path = '/home/gibe/mslearn/simpleCNN-label/2.png'

# Load and preprocess the image
img_array = load_and_preprocess_image(img_path)

# Make a prediction
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions, axis=1)

# Get the class names
class_names = ['blueplayers', 'others']  # Replace with your actual class names

# Print the predicted class
print(f'The model predicts this image is a: {class_names[predicted_class[0]]}')
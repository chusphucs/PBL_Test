from keras.models import load_model
from keras.preprocessing import image
import numpy as np
import keras

from train import classes

# Load model
model = load_model('models/model.h5')

# Chuẩn bị ảnh
img_path = 'MANTRAU.jpg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array = keras.applications.mobilenet.preprocess_input(img_array)

# Dự đoán
predictions = model.predict(img_array)
predicted_class = np.argmax(predictions[0])

# In kết quả
print("Predicted Class:", classes[predicted_class])

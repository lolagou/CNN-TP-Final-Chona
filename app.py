import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image
import numpy as np

model = load_model("modeloChona.h5") 


def preprocess_image(image):
    image = image.resize((128, 128))  
    image = np.array(image) / 255.0
    return np.expand_dims(image, axis=0)

st.title("Clasificador de imágenes de gatos y perros")
st.write("Sube una imagen para clasificar si es un gato o un perro.")

uploaded_file = st.file_uploader("Elige una imagen...", type=["jpg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Imagen cargada.', use_column_width=True)
    
    st.write("Clasificando...")
    preprocessed_image = preprocess_image(image)
    prediction = model.predict(preprocessed_image)
    
    # Ajusta según la salida de tu modelo (ej. softmax: gato=0, perro=1)
    class_names = ['Gato', 'Perro']
    predicted_class = class_names[int(prediction[0][0] > 0.5)]
    
    st.write(f'Predicción: **{predicted_class}**')
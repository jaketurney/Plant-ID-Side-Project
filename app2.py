# For the new model I built on 1/16/2021. Has all 12 plants! Validation Accuracy of 64%

### So we want to build a web app that you can upload a photo to and our model predicts its classification.
### To do this I found this video:
# https://www.youtube.com/watch?v=Q1NC3NbmVlc

# First we want to save our model.
#tf.keras.models.save_model(model,"plant_model.hdf5")

# Install streamlit
#pip install streamlit

import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

# Make a function that loads up the saved model file (plat_model.hdf5)
def load_model():
    mdl = tf.keras.models.load_model(r"C:\Users\jettu\OneDrive\Documents\Plant Images\plant_model2.hdf5")
    return mdl
mdl = load_model()

st.write("""
         # Plant Classification
         """
         )

file = st.file_uploader("Please upload a plant image", type=["jpg", "png"])

import cv2
from PIL import Image, ImageOps
import numpy as np

def import_and_predict(image_data, model):
    
    size = (224,224)
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img = np.asarray(image)
    img_reshape = img[np.newaxis,...]
    prediction = model.predict(img_reshape)
    
    return prediction

if file is None:
    st.text("Please Upload an Image File")
else: 
    image = Image.open(file)
    st.image(image, use_column_width=True)
    predictions = import_and_predict(image, mdl)
    class_names = ["black_grass", "charlock", 'cleavers', 'common chickweed', 'common wheat', 'fat hen', 'loose silky-bent', 'maize', 'scentless mayweed', 'shepherds purse', 'small-flowered cranesbill', 'sugar beet']
    string="This image is most likely a" + class_names[np.argmax(predictions)]
    st.success(string)

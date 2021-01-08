#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D , MaxPool2D , Flatten , Dropout 
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

import cv2
import os

import numpy as np


# In[7]:


os.getcwd()
#os.chdir(r"C:\Users\jettu\OneDrive\Documents\Plant Images")


# In[8]:


get_ipython().run_cell_magic('writefile', 'app.py', '### So we want to build a web app that you can upload a photo to and our model predicts its classification.\n### To do this I found this video:\n# https://www.youtube.com/watch?v=Q1NC3NbmVlc\n\n# First we want to save our model.\n#tf.keras.models.save_model(model,"plant_model.hdf5")\n\n# Install streamlit\n#pip install streamlit\n\nimport streamlit as st\nimport tensorflow as tf\n\nst.set_option(\'deprecation.showfileUploaderEncoding\', False)\n@st.cache(allow_output_mutation=True)\n\n# Make a function that loads up the saved model file (plat_model.hdf5)\ndef load_model():\n    mdl = tf.keras.models.load_model(r"C:\\Users\\jettu\\OneDrive\\Documents\\Plant Images\\plant_model.hdf5")\n    return mdl\nmdl = load_model()\n\nst.write("""\n         # Plant Classification\n         """\n         )\n\nfile = st.file_uploader("Please upload a plant image", type=["jpg", "png"])\n\nimport cv2\nfrom PIL import Image, ImageOps\nimport numpy as np\n\ndef import_and_predict(image_data, model):\n    \n    size = (224,224)\n    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)\n    img = np.asarray(image)\n    img_reshape = img[np.newaxis,...]\n    prediction = model.predict(img_reshape)\n    \n    return prediction\n\nif file is None:\n    st.text("Please Upload an Image File")\nelse: \n    image = Image.open(file)\n    st.image(image, use_column_width=True)\n    predictions = import_and_predict(image, mdl)\n    class_names = ["black_grass", "charlock"]\n    string="This image is most likely a " + class_names[np.argmax(predictions)]\n    st.success(string)')


# The code above is for building the app using streamlit and tensorflow. The app is then saved and written into a file
# All code below should be used in a Python terminal from Anaconda

# In[3]:


ngrok authtoken 1mfEEb8I2XKI1HcegrTvlBLBMQr_6etgoH4zNdXkykMKrB6SR
# May not need this for the terminal. Not too sure what this is really for.


# In[19]:


pip install streamlit #incase it has not been in the terminal
streamlit hello #tests to make sure streamlit is working (can skip)
streamlit run app.py #runs the app file and opens a web page
# ctrl c in terminal to stop the application


# In[22]:


# This was in that video 
#pip install pyngrok
from pyngrok import ngrok 
url=ngrok.connect(port=80)
url


# In[ ]:





# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 16:49:55 2021

@author: jettu
"""

# Source of walkthrough: 
# https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/


### STEP 1: Import all of the packages we need
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

os.chdir("C:/Users/jettu/OneDrive/Documents/Plant Images/")
os.getcwd()

#### Step 2: Next, we will read in the data ans split into training and validation
labels = ['black_grass', 'charlock', 'cleavers', 'common chickweed', 'common wheat', 'fat hen', 'loose silky-bent', 'maize', 'scentless mayweed', 'shepherds purse', 'small-flowered cranesbill', 'sugar beet']
img_size = 224
def get_data(data_dir):
    data = [] # Create an empty list
    for label in labels: 
        path = os.path.join(data_dir, label) # writes the path using the directory you input
        class_num = labels.index(label) # Identifies the image using the index of the label in labels
        for img in os.listdir(path):
            try:
                img_arr = cv2.imread(os.path.join(path, img))[...,::-1] #convert BGR to RGB format
                resized_arr = cv2.resize(img_arr, (img_size, img_size)) # Reshaping images to preferred size
                data.append([resized_arr, class_num])
            except Exception as e:
                print(e)
    return np.array(data)

#Now we can easily fetch our train and validation data from the folders that we made.
train = get_data("C:/Users/jettu/OneDrive/Documents/Plant Images/Train") # a 0 indicates blackgrass
val = get_data("C:/Users/jettu/OneDrive/Documents/Plant Images/Valid")

#### Step 3: Visualize the data set using the seaborn package
l = []
for i in train:
    if(i[1] == 0):
        l.append("black_grass")
    elif(i[1]==1):
        l.append("charlock")
    elif(i[1]==2):
        l.append("cleavers")
    elif(i[1]==3):
        l.append("common chickweed")
    elif(i[1]==4):
        l.append("common wheat")
    elif(i[1]==5):
        l.append("fat hen")
    elif(i[1]==6):
        l.append("loose silky-bent")
    elif(i[1]==7):
        l.append("maize")
    elif(i[1]==8):
        l.append("scentless mayweed")
    elif(i[1]==9):
        l.append("shepherds purse")
    elif(i[1]==10):
        l.append("small-flowered cranesbill")
    else:
        l.append("sugar beet")
sns.set_style('darkgrid')
sns.countplot(l)

# Now we can go ahead and plot a random photo to take a peak at it and make sure all is working
# Plot a black_grass image
plt.figure(figsize = (5,5))
plt.imshow(train[1][0])
plt.title(labels[train[0][1]])

# Plot a charlock
plt.figure(figsize = (5,5))
plt.imshow(train[201][0])
plt.title(labels[train[201][1]])

# Plot a sugar beet 
plt.figure(figsize = (5,5))
plt.imshow(train[2390][0])
plt.title(labels[train[2390][1]])

#### Step 4: Data preprocessing and augmentation
x_train = []
y_train = []
x_val = []
y_val = []

for feature, label in train:
  x_train.append(feature) # the features are the arrays of information that tell the computer what the image is
  y_train.append(label) # the labels are the classifications of the images

for feature, label in val:
  x_val.append(feature)
  y_val.append(label)

# Normalize the data. We use 255 because Images are 3-dimensional arrays of integers from 0 to 255, of size Width x Height x 3
x_train = np.array(x_train) / 255 # This standardizes the data
x_val = np.array(x_val) / 255 # Also standardizes the data

x_train.reshape(-1, img_size, img_size, 1)
y_train = np.array(y_train)

x_val.reshape(-1, img_size, img_size, 1)
y_val = np.array(y_val)

# Data augmentation
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range = 50,  # randomly rotate images in the range (degrees, 0 to 180) [30,50]
        zoom_range = 0.2, # Randomly zoom image 
        width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
        horizontal_flip = True,  # randomly flip images
        vertical_flip=True)  # randomly flip images [False, True]


datagen.fit(x_train)


#### Step 5: Define the model. 
# "Let’s define a simple CNN model with 3 Convolutional layers followed by max-pooling 
# layers. A dropout layer is added after the 3rd maxpool operation to avoid overfitting."

model = Sequential()
model.add(Conv2D(32, 3,padding="same", activation="relu", input_shape=(224,224,3)))
model.add(MaxPool2D())

model.add(Conv2D(64, 3, padding="same", activation="relu")) # Changed 32 to 64, 1/16/2021
model.add(MaxPool2D())


model.add(Conv2D(64, 3, padding="same", activation="relu"))
model.add(MaxPool2D())
model.add(Dropout(0.4))

model.add(Flatten()) 
model.add(Dense(128,activation="relu"))
model.add(Dense(12, activation="softmax")) # had to chance the 2 to a 12 so that we included all 11 possible labels

model.summary()

## Let’s compile the model now using Adam as our optimizer and SparseCategoricalCrossentropy
## as the loss function. We are using a lower learning rate of 0.000001 for a smoother curve.

opt = Adam(lr=0.001) #Original was 0.000001, 0.0001
model.compile(optimizer = opt , loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True) , metrics = ['accuracy'])

## Now, let’s train our model for 500 epochs since our learning rate is very small.

history = model.fit(x_train,y_train,epochs = 15 , validation_data = (x_val, y_val))

#### Step 6: Evaluate the results
# we can make a few plots that show the results of our training over time
# The plots also show the increased accuracy over time for training and validation

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(15) #Change to the number of epochs you used

plt.figure(figsize=(15, 15))
plt.subplot(2, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(2, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()

# Check the accuracy statistics. We can look at the precision and recall as well.
predictions = model.predict_classes(x_val)
predictions = predictions.reshape(1,-1)[0]
print(classification_report(y_val, predictions, target_names = ['Black Grass (Class 0)','Charlock (Class 1)', 'Cleavers (Class 2)', 'Common Chickweed (Class 3)', 'Common Wheat (Class 4)', 'Fat Hen (Class 5)', 'Loose Silky-Bent (Class 6)', 'Maize (Class 7)', 'Scentless Mayweed (Class 8)', 'Shepherds Purse (Class 9)', 'Small-Flowered Cranesbill (Class 10)', 'Sugar Beet(Class 11)']))

# Our model prdicted 6 of the 100 images wrong for its classification. Lets look at one image it got wrong:
plt.figure(figsize = (5,5))
plt.imshow(val[31][0])
plt.title(labels[val[31][1]])

# Test on new data we do not know the answer to:
test = get_data("C:/Users/jettu/OneDrive/Documents/Plant Images/test")
train = get_data("C:/Users/jettu/OneDrive/Documents/Plant Images/Train") # a 0 indicates blackgrass

testing_x = []
testing_y = []

for feature, label in test:
  testing_x.append(feature)
  testing_y.append(label)
  
testing_x = np.array(testing_x) / 255

testing_x.reshape(-1, img_size, img_size, 1)

test_predictions = model.predict_classes(testing_x)
test_predictions = test_predictions.reshape(1,-1)[0]

#### Next step: Build a function using the test code above so that any new photo that 
# is uploaded, it can go through the model and get a prediction.

### So we want to build a web app that you can upload a photo to and our model predicts its classification.
### To do this I found this video:
# https://www.youtube.com/watch?v=Q1NC3NbmVlc

# First we want to save our model.
tf.keras.models.save_model(model,"plant_model2.hdf5")

# Validation accuracy: 0.69
# Training accuracy: 0.97

# Install streamlit
pip install streamlit

#%% BELOW HERE WILL WORK IN A JUPYTER NOTEBOOK

%%writefile app.py

import streamlit as st
import tensorflow as tf

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)

# Make a function that loads up the saved model file (plat_model.hdf5)
def load_model():
    mdl = tf.keras.models.load_model(r"C:\Users\jettu\OneDrive\Documents\Plant Images\plant_model.hdf5")
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
    predictions = import_and_predict(image, model)
    class_names = ["black_grass", "charlock"]
    string="This image is most likely a" + class_names[np.argmax(predictions)]
    st.success(string)

#%%




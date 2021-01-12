# Plant-ID-Side-Project
To better understand neural networks and image classification, I downloaded a plant photo data set from Kaggle and built out a classification model in Python. I then created an interactive web app where a user can upload a picture of a plant and the model will make a prediction on its classification. 

Here we have a walk through of how to:
1. Build an image classification model within Spyder or Jupyter Notebooks
2. Use streamlit to build an app of the model in Jupyter Notebooks
3. Deploy the app of the model built with streamlit using the CMD Terminal 

Kaggle link to data: https://www.kaggle.com/vbookshelf/v2-plant-seedlings-dataset

First, build the model. I used this resource for an image classification model using a convolutional nueral network:
https://www.analyticsvidhya.com/blog/2020/10/create-image-classification-model-python-keras/

I performed all of the code from above in Spyder. I then saved my model at the end once it was finalized into a script.

Once I have our model saved, I move onto Jupyter Notebooks.

Second, I go into Jupyter Notebook and code the app using streamlit. 
This video was so helpful in building the app: https://www.youtube.com/watch?v=Q1NC3NbmVlc
Another great resource: https://towardsdatascience.com/how-to-build-an-image-classification-app-using-logistic-regression-with-a-neural-network-mindset-1e901c938355

Save the app as app_name.py

NEED TO MAKE SURE of a few things:
- the working directory of the notebook is where the app.py is located
- the ngrok application is where the app.py is (so same directory)


Thirdly, I open a Terminal through Anaconda. Open Anaconda, go to Environments, Click the Arrow next to Python3, Click Open Terminal.
A terminal for Python3 will open. You will need to code this:
1. cd "the working directory path"
2. pip install streamlit
3. streamlit hello (to make sure streamlit is working)
4. cntrl c to close
5. *May not need this step* ngrok authtoken 1mfEEb8I2XKI1HcegrTvlBLBMQr_6etgoH4zNdXkykMKrB6SR (your ngrok key to connect to the internet)
6. streamlit run app.py

It should open up automatically to where you want to go!

Next steps:
- Add more photos of different species of weeds to the model so it can classify other types of weeds
- Scrape images from google on the different weeds to build a more accurate model

# -*- coding: utf-8 -*-
"""
Created on Fri Jun 26 00:40:50 2020
@author: ASUS
"""
import os
import cv2
import pandas as pd
import numpy as np
import keras
import pickle
import numpy as np
import streamlit as st
import tensorflow as tf
from PIL import Image, ImageOps

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array 

filename = 'new_model.h5'
#filename = 'model.sav'

#with open(filename, "rb") as f:
#    rawdata = f.read()

#model = pickle.loads(rawdata)

#model = pickle.load(open(filename, 'rb'))


model = load_model(filename)





st.write("""
         # Breast Cancer Prediction
         """
         )

#st.write("This is a simple image classification web app to predict rock-paper-scissor hand sign")

file = st.file_uploader("Please upload an image file", type=["jpg", "png"])
#
if file is None:
    st.text("You haven't uploaded an image file")
else:
    #file_details = {"FileName":file.name,"FileType":file.type}
    #st.write(file_details)
    #img = load_img(file)
    image = Image.open(file)
    #file = str(file)
    #image = []
    #image.append(cv2.imread(img))
    st.image(image, use_column_width=True)

    #image = np.array(image)
    #image = image.astype('float32') / 255
    #class_int = import_and_predict(image, model)
    #prediction = model.predict(image)
    #class_int = np.argmax(prediction,axis=-1)

    
    size = (100,100)    
    image = ImageOps.fit(image, size, Image.ANTIALIAS)
    image = image.convert('RGB')
    image = np.asarray(image)
    image = (image.astype(np.float32) / 255.0)

    img_reshape = image[np.newaxis,...]
    st.write(img_reshape)
 
    prediction = model.predict(img_reshape)
        
        
    
    st.write(prediction)
 
    if (np.argmax(class_int)==0):
       st.write("Density1Benign")
    elif (np.argmax(class_int)==1):  
       st.write("Density1Malignant")
    elif (np.argmax(class_int)==2):
       st.write("Density2Benign")
    elif (np.argmax(class_int)==3):
       st.write("Density2Malignant")
    elif (np.argmax(class_int)==4):
       st.write("Density3Benign")
    elif (np.argmax(class_int)==5):
       st.write("Density3Malignant")
    elif (np.argmax(class_int)==6):
       st.write("Density4Benign")
    elif (np.argmax(class_int)==7):
       st.write("Density4Malignant")

    st.write(class_int)


import numpy as np
import pickle
import streamlit as st
import tensorflow as tf
from PIL import Image

loaded_model = pickle.load(open('C:/Users/Sindhu/Jupyter/trained_model.sav','rb'))

# creating a function for Prediction

def rockmine_prediction(inp_img):
    

    img = Image.open(inp_img)

    resize = tf.image.resize(img, (256,256))

    np.expand_dims(resize, 0)

    yhat = loaded_model.predict(np.expand_dims(resize/255, 0))

    if yhat > 0.5:
        return ('Rock')
    else:
        return ('Mine')
  
    
  
def main():
    
    
    # giving a title
    st.title('Mine and Rock Prediction Web App')
    
    
    # getting the input data from the user
    
    input_image_path = st.file_uploader("Pick a file")
    #input_image_path = "C:/Users/Sindhu/OneDrive/Desktop/Jupyter/mine_test.jpg"
    
    
    # code for Prediction
    prediction_val = ''
    
    # creating a button for Prediction
    
    if st.button('Result'):
        prediction_val = rockmine_prediction(input_image_path)
        
        
    st.success(prediction_val)
    
    
    
    
    
if __name__ == '__main__':
    main()


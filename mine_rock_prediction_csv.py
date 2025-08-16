import numpy as np
import pickle
import streamlit as st


loaded_model = pickle.load(open('C:/Users/Sindhu/Jupyter/csv_trained_model.sav','rb'))


def csv_pred(input_data):


    # changing the input_data to a numpy array
    input_data_as_numpy_array = np.asarray(input_data)

    # reshape the np array as we are predicting for one instance
    input_data_reshaped = input_data_as_numpy_array.reshape(1,-1)

    prediction = loaded_model.predict(input_data_reshaped)

    if (prediction[0]=='R'):
      return 'The object is a Rock'
    else:
      return 'The object is a mine'
    
    
def main():
    st.title('CSV Mine or Rock Prediction')
    
    input_numbers = st.text_area("Enter a list of numbers (comma-separated):")

    # Parse the input to create a list of numbers
    signal_val = [float(num.strip()) for num in input_numbers.split(',') if num.strip()]

    
    #signal_val = st.text_input('Values of sonar data');
    
    res = ''
    
    if st.button('Result'):
        res = csv_pred([signal_val])
        
    st.success(res)
    
    
if __name__ == '__main__':
    main()
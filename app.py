import pickle
import numpy as np 
import pandas as pd 
import streamlit as st

with open('model.pkl', 'rb') as file:
    model=pickle.load(file)

with open('preprocessor.pkl', 'rb') as file:
    preprocessor=pickle.load(file)

st.title('Penguin Classifier')

island = st.selectbox('island', ['Torgersen', 'Biscoe', 'Dream'])
bill_length_mm = st.slider('bill_length_mm', 40, 55)
bill_depth_mm = st.slider('bill_depth_mm', 15, 20)
flipper_length_mm = st.slider('flipper_length_mm', 175, 230)
body_mass_g = st.slider('body_mass_g', 2700, 6300)
sex = st.selectbox('sex', ['Male', 'Female'])

input_data = pd.DataFrame({
    'island': [island],
    'bill_length_mm': [bill_length_mm],
    'bill_depth_mm': [bill_depth_mm],
    'flipper_length_mm': [flipper_length_mm],
    'body_mass_g': [body_mass_g],
    'sex': [sex]
})

processed_input_data = preprocessor.transform(input_data)
prediction = model.predict(processed_input_data)

st.write(f'Penguin species: {prediction}')

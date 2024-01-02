import pandas as pd
import joblib
import streamlit as st
import pickle
import os
os.environ['PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION'] = 'python'


# File path to the saved model
model_file_path = "./model_folder/model1.pkl"

# Load the model
model = joblib.load(model_file_path)
# UI Elements to get input
st.title('Disease Prediction')
Age = st.number_input('Age', step = 1)
Gender = st.text_input("Gender (Male OR Female)")
BP = st.number_input('BP', step = 1)
Smoker_Status = st.text_input("Smoker_Status ( Smoker OR Not_Smoker)")
# Make as much input as much we want

# Format the input data into a DataFrame
new_data = pd.DataFrame({'Age':Age, 'Gender': Gender, 'BP': BP, 'Smoker_Status': Smoker_Status}, index=[0])

result = ""
if st.button("Predict"):
    result = model.predict(new_data)

    if result == 0:
        output = 'Patient have Disease'
    else:
        output = 'Patient have No Disease'
    st.subheader("Results")
    st.subheader(output)
else:
    print('Fill the values in the above boxes and hit on the predict button')

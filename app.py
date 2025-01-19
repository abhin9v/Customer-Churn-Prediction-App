import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# Correct import statement
from tensorflow.keras.models import load_model

# Load the model
model = load_model('model.h5')

with open('onehotencoder.pkl', 'rb') as file:
    onehotencoder = pickle.load(file)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('scale.pkl', 'rb') as file:
    scale = pickle.load(file)

st.title('Customer Churn Prediction')

geography = st.selectbox('Geography', onehotencoder.categories_[0])
gender = st.selectbox('Gender', label_encoder_gender.classes_)

credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=5)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credit Card', [0, 1])
is_active_member = st.selectbox('Is Active Member', [0, 1])
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=50000.0)

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Geography': [geography],
    'Gender': [gender],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

# Encode Geography
geo_encoder = onehotencoder.transform([[input_data['Geography'][0]]]).toarray()
geo_encoder_df = pd.DataFrame(geo_encoder, columns=onehotencoder.get_feature_names_out(['Geography']))

# Concatenate encoded Geography and drop original column
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoder_df], axis=1)
input_data = input_data.drop(columns=['Geography'])

# Encode Gender
input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

# Ensure all columns are numeric before scaling
input_data = input_data.apply(pd.to_numeric)

# Scale input data
input_data_scaled = scale.transform(input_data)

# Predict
prediction = model.predict(input_data_scaled)
prediction_proba = (prediction[0][0]) * 100

st.write(f"Churn Probability: {prediction_proba:.2f}")

if prediction_proba > 50:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')
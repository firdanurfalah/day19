import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.write("""
# DATASET HEART
""")

st.sidebar.header('User Input Features')



# Collects user input features into dataframe

def user_input_features():
    Age = st.sidebar.number_input('Enter your age: ')

    Sex  = st.sidebar.selectbox('Sex',(0,1))
    ChestPainType = st.sidebar.selectbox('Chest pain type',(0,1,2,3))
    RestingBP = st.sidebar.number_input('Resting blood pressure: ')
    Cholesterol = st.sidebar.number_input('Serum cholestoral in mg/dl: ')
    FastingBS = st.sidebar.selectbox('Fasting blood sugar',(0,1))
    RestingECG = st.sidebar.number_input('Resting electrocardiographic results: ')
    MaxHR = st.sidebar.number_input('Maximum heart rate achieved: ')
    ExerciseAngina = st.sidebar.selectbox('Exercise induced angina: ',(0,1))
    Oldpeak = st.sidebar.number_input('oldpeak ')
    ST_Slope = st.sidebar.number_input('he slope of the peak exercise ST segmen: ')
    ca = st.sidebar.selectbox('number of major vessels',(0,1,2,3))
    thal = st.sidebar.selectbox('thal',(0,1,2))

    data = {'Age': Age,
            'Sex': Sex, 
            'ChestPainType': ChestPainType,
            'RestingBP':RestingBP,
            'Cholesterol': Cholesterol,
            'FastingBS': FastingBS,
            'RestingECG': RestingECG,
            'MaxHR':MaxHR,
            'ExerciseAngina':ExerciseAngina,
            'Oldpeak':Oldpeak,
            'ST_Slope':ST_Slope,
            'ca':ca,
            'thal':thal
                }
    features = pd.DataFrame(data, index=[0])
    return features
input_df = user_input_features()

# Combines user input features with entire dataset
# This will be useful for the encoding phase
heart_dataset = pd.read_csv('heart.csv')
heart_dataset = heart_dataset.drop(columns=['HeartDisease'])

df = pd.concat([input_df,heart_dataset],axis=0)

# Encoding of ordinal features
# https://www.kaggle.com/pratik1120/penguin-dataset-eda-classification-and-clustering
df = pd.get_dummies(df, columns = ['Sex', 'ChestPainType', 'FastingBS', 'RestingECG', 'ExerciseAngina', 'ST_Slope', 'ca', 'thal'])

df = df[:1] # Selects only the first row (the user input data)

st.write(input_df)
# Reads in saved classification model
load_clf = pickle.load(open('DAY19/Random_forest_model.pkl', 'rb'))

# Apply model to make predictions
prediction = load_clf.predict(df)
prediction_proba = load_clf.predict_proba(df)


st.subheader('Prediction')
st.write(prediction)

st.subheader('Prediction Probability')
st.write(prediction_proba)
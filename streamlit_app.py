streamlit_app.py
import streamlit as st
import pandas as pd
import joblib

# تحميل الموديل
model = joblib.load('sales_forecasting_model.pkl')

st.title("Sales Forecasting App")
st.write("Welcome to the prediction app!")

# خانات إدخال البيانات
val1 = st.number_input("Feature 1 Value", value=0.0)
val2 = st.number_input("Feature 2 Value", value=0.0)

if st.button("Predict"):
    # تجهيز البيانات
    input_data = pd.DataFrame([[val1, val2]], columns=['feature_1', 'feature_2'])
    
    # التوقع
    prediction = model.predict(input_data)
    
    st.success(f"The Predicted Result is: {prediction[0]:.2f}")
    st.balloons()
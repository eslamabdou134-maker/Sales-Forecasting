import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Page Configuration
st.set_page_config(page_title="Sales Forecaster Pro", page_icon="📈", layout="centered")

# 2. Model Loading
@st.cache_resource
def load_model():
    return joblib.load('sales_forecasting_model.pkl')

model = load_model()

# 3. User Interface
st.title("📈 Sales Forecasting System")
st.markdown("---")
st.write("Enter the required parameters below to generate a sales prediction.")

# Organizing inputs into columns
col1, col2 = st.columns(2)

with col1:
    val1 = st.number_input("Unit Price", value=10.0, help="Enter the product price")
    
with col2:
    val2 = st.number_input("Marketing Budget", value=20.0, help="Enter the advertising spend")

# 4. Data Visualization (Optional Placeholder)
st.markdown("### 📊 Sales Trends Analysis")
chart_data = pd.DataFrame(
    np.random.randn(20, 2),
    columns=['Actual Sales', 'Predicted Sales']
)
st.line_chart(chart_data)

# 5. Prediction Logic
if st.button("Generate Prediction"):
    # Reshape input for the model
    input_data = np.array([[val1, val2]])
    prediction = model.predict(input_data)
    
    st.success(f"### Predicted Result: {prediction[0]:.2f}")
    st.balloons()

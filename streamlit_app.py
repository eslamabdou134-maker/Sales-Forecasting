import streamlit as st
import pandas as pd
import joblib
import datetime

# تحميل النموذج
@st.cache_resource
def load_model():
    return joblib.load('sales_forecasting_model.pkl')

model = load_model()

st.title("لوحة توقع مبيعات التجزئة")

# مدخلات المستخدم
target_date = st.date_input("اختر التاريخ للتوقع", datetime.date(2017, 8, 16))
onpromotion = st.selectbox("هل يوجد عرض ترويجي؟", ["No", "Yes"])

if st.button("عرض التوقعات"):
    # تحضير البيانات للنموذج
    input_data = pd.DataFrame({
        'onpromotion': [1 if onpromotion == "Yes" else 0],
        'year': [target_date.year],
        'month': [target_date.month],
        'day': [target_date.day],
        'dayofweek': [target_date.weekday()]
    })
    
    prediction = model.predict(input_data)
    st.success(f"حجم المبيعات المتوقع: {prediction[0]:,.2f}")

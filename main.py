# This is a sample Python script.
#import pickle
import joblib as jl
import pandas as pd
import streamlit as st

st.header("Prediction of 28-day mortality among orthopaedic trauma patients admitted to ICU using machine learning techniques.")
st.sidebar.title("Parameters Selection Panel")
st.sidebar.markdown("Picking up parameters")

Shockbloodloss = st.sidebar.selectbox("Hemorrhagic shock", ("No", "Yes"))
Temperature = st.sidebar.slider("Temperature (℃)", 34.0, 39.0)
Respiratoryrate = st.sidebar.slider("Respiratory rate (BMP)", 12, 38)
BUNday1 = st.sidebar.slider("Blood urea nitrogen (mmol/L)", 2.40, 38.00)
Kday1 = st.sidebar.slider("Potassium (mmol/L)", 3.15, 5.85)
Redbloodcelcount = st.sidebar.slider("Red blood cell count(×10^12/L)", 1.55, 5.90)
GCS = st.sidebar.slider("GCS", 3, 15)

if st.button("Submit"):
    rf_clf = jl.load("Xgbc_clf_final_round.pkl")
    x = pd.DataFrame([[Shockbloodloss, Temperature, Respiratoryrate, BUNday1, Kday1, Redbloodcelcount, GCS]],
                     columns=["Shockbloodloss", "Temperature", "Respiratoryrate", "BUNday1", "Kday1", "Redbloodcelcount", "GCS"])
    x = x.replace(["No", "Yes"], [0, 1])

    # Get prediction
    prediction = rf_clf.predict_proba(x)[0, 1]
        # Output prediction
    st.text(f"Probability of severe sleep disturbance: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.254:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.254:
        st.markdown(f"Recommendation: Routine preoperative evaluation and management with XXX.")
    else:
        st.markdown(f"xxxxxx.")
st.subheader('Model information')
st.markdown('The XGBoosting machine algorithm was employed to construct the model, yielding an outstanding area under the curve (AUC) value of 0.982 (95%CI: 0.976-0.990). This web-based calculator has been intricately designed to accurately assess the likelihood of 28-day mortality in individuals with orthopaedic trauma who have been admitted to the intensive care unit (ICU). Notably, this tool is effortlessly accessible at absolutely no expense and is exclusively intended for the purpose of advancing research endeavors.')
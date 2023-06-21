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
    st.text(f"Probability of 28-day mortality: {'{:.2%}'.format(round(prediction, 5))}")
    if prediction < 0.546:
        st.text(f"Risk group: low-risk group")
    else:
        st.text(f"Risk group: High-risk group")
    if prediction < 0.546:
        st.markdown(f"Recommendation: Regular monitoring and observation: Although low-risk trauma patients have a lower likelihood of early mortality, they still require careful monitoring and observation. Regular assessment of vital signs, pain levels, and wound healing can help detect any signs of deterioration or complications. Early mobilization and rehabilitation: Initiating early mobilization and rehabilitation in low-risk trauma patients can help prevent complications and promote faster recovery. This includes physical therapy, occupational therapy, and psychological support to enhance functional outcomes and quality of life.")
    else:
        st.markdown(f"Recommendation: Timely resuscitation and stabilization: Immediate resuscitation and stabilization are critical for high-risk trauma patients. Prompt administration of intravenous fluids, blood products, and medications to maintain adequate perfusion and oxygenation can help prevent further complications and improve outcomes. Multidisciplinary care: High-risk trauma patients require comprehensive and multidisciplinary care. This includes close collaboration between trauma surgeons, intensivists, nurses, respiratory therapists, and other healthcare professionals. Regular multidisciplinary team meetings can ensure coordinated and optimal care.")
st.subheader('Model information')
st.markdown('The XGBoosting machine algorithm was employed to construct the model, yielding an outstanding area under the curve (AUC) value of 0.982 (95%CI: 0.976-0.990). This web-based calculator has been intricately designed to accurately assess the likelihood of 28-day mortality in individuals with orthopaedic trauma who have been admitted to the intensive care unit (ICU). Notably, this tool is effortlessly accessible at absolutely no expense and is exclusively intended for the purpose of advancing research endeavors.')

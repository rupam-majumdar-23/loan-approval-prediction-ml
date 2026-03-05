import streamlit as st
import numpy as np
import pickle

model = pickle.load(open("model.pkl", "rb"))


st.title("🏦 Loan Approval Prediction System")
st.write("Enter applicant details to check loan approval status")


st.sidebar.header("Applicant Details")

income = st.sidebar.number_input("Income", min_value=0)
credit_score = st.sidebar.number_input("Credit Score", min_value=0)
loan_amount = st.sidebar.number_input("Loan Amount", min_value=0)
years_employed = st.sidebar.number_input("Years Employed", min_value=0)
points = st.sidebar.number_input("Bank Points", min_value=0)

if st.button("Predict Loan Status"):

    input_data = np.array([income, credit_score, loan_amount, years_employed, points]).reshape(1, -1)

    prediction = model.predict(input_data)
    probability = model.predict_proba(input_data)

    st.subheader("Prediction Result")

    if prediction[0] == 1:
        st.success("✅ Loan Approved")
        st.balloons()
    else:
        st.error("❌ Loan Rejected")


    st.write("Approval Probability:", round(probability[0][1]*100,2), "%")

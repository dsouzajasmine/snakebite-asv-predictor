import streamlit as st
import pandas as pd
import joblib
import requests

import firebase_admin
from firebase_admin import credentials, firestore, auth

from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet

from datetime import datetime

# -------------------------------
# Session State
# -------------------------------
if "page" not in st.session_state:
    st.session_state.page = "login"

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "user_email" not in st.session_state:
    st.session_state.user_email = ""

if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "user_id" not in st.session_state:
    st.session_state.user_id = ""

model = joblib.load('asv_model.pkl')
feature_list = joblib.load('model_features.pkl')

# Initialize Firebase only once
if not firebase_admin._apps:
    # Check if running on Streamlit Cloud (secrets) or locally (json file)
    try:
        # Streamlit Cloud — read from secrets
        firebase_cred = dict(st.secrets["firebase_key"])
        cred = credentials.Certificate(firebase_cred)
    except:
        # Local VS Code — read from json file
        cred = credentials.Certificate('firebase_key.json')
    
    firebase_admin.initialize_app(cred)

db = firestore.client()

def login_page():

    st.title("Sakebite ASV Severity Predictor")
    st.caption("AI-Based Clinical Decision Support System")

    st.subheader("Login")

    email = st.text_input("Email")

    password = st.text_input("Password", type="password")

    if st.button("Login"):

        response = login_user(email, password)

        if response.status_code == 200:

            user = auth.get_user_by_email(email)

            st.session_state.logged_in = True
            st.session_state.user_email = email
            st.session_state.user_name = user.display_name
            st.session_state.user_id = user.uid
            st.session_state.page = "predict"
            
            st.success(f"Welcome, {user.display_name}!")

            st.rerun()

        else:

            st.error("Invalid Email or Password")

    st.write("Don't have an account?")

    if st.button("Register"):
        st.session_state.page = "register"
        st.rerun()

def login_user(email, password):
    try:
        api_key = st.secrets["firebase_key"]["api_key"]
    except:
        api_key = "your-local-api-key"  # for local testing
    
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={api_key}"
    payload = {
        "email": email,
        "password": password,
        "returnSecureToken": True
    }
    response = requests.post(url, json=payload)
    return response

def register_page():

    st.title("Snakebite ASV Severity Predictor")
    st.caption("AI-Based Clinical Decision Support System")

    st.subheader("Create Account")

    name = st.text_input("Full Name")

    email = st.text_input("Email")

    password = st.text_input("Password", type="password")

    confirm_password = st.text_input("Confirm Password", type="password")

    if st.button("Create Account"):

        # Check if all fields are filled
        if name == "" or email == "" or password == "" or confirm_password == "":
            st.error("Please fill all fields.")

        # Check if passwords match
        elif password != confirm_password:
            st.error("Passwords do not match!")

        else:
            try:
                # Create user in Firebase Authentication
                user = auth.create_user(
                    email=email,
                    password=password,
                    display_name=name
                )

                # Save user details in Firestore
                db.collection("users").document(user.uid).set({
                    "name": name,
                    "email": email
                })

                st.success("Account created successfully!")
                st.info("Please go back and login.")

            except Exception as e:
                st.error(f"Registration failed: {e}")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

def create_pdf(patient_name, gender, age_group, snake,
               hb, platelets, pt, inr, aptt, urea,
               creatinine, sgot, sgpt, crp,
               severity, asv):

    pdf_file = "Snakebite_Report.pdf"

    doc = SimpleDocTemplate(pdf_file)
    styles = getSampleStyleSheet()

    elements = []

    elements.append(Paragraph("<b>SNAKEBITE ASV SEVERITY REPORT</b>", styles["Title"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(Paragraph("<b>PATIENT DETAILS</b>", styles["Heading2"]))
    if patient_name.strip():
        elements.append(Paragraph(f"<b>Patient Name:</b> {patient_name}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Gender:</b> {gender}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Age Group:</b> {age_group}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Snake:</b> {snake}", styles["Normal"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(Paragraph("<b>LAB SUMMARY</b>", styles["Heading2"]))
    elements.append(Paragraph(f"<b>Hemoglobin (Hb):</b> {hb}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Platelets:</b> {platelets}", styles["Normal"]))
    elements.append(Paragraph(f"<b>PT:</b> {pt}", styles["Normal"]))
    elements.append(Paragraph(f"<b>INR:</b> {inr}", styles["Normal"]))
    elements.append(Paragraph(f"<b>aPTT:</b> {aptt}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Urea:</b> {urea}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Creatinine:</b> {creatinine}", styles["Normal"]))
    elements.append(Paragraph(f"<b>SGOT:</b> {sgot}", styles["Normal"]))
    elements.append(Paragraph(f"<b>SGPT:</b> {sgpt}", styles["Normal"]))
    elements.append(Paragraph(f"<b>CRP:</b> {crp}", styles["Normal"]))

    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(Paragraph("<b>PREDICTION RESULT</b>", styles["Heading2"]))
    elements.append(Paragraph(f"<b>Predicted Severity:</b> {severity}", styles["Normal"]))
    elements.append(Paragraph(f"<b>Recommended ASV:</b> {asv}", styles["Normal"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))

    current_datetime = datetime.now().strftime("%d %B %Y %I:%M %p")
    elements.append(Paragraph(f"<b>Generated On:</b> {current_datetime}", styles["Normal"]))

    elements.append(Paragraph("<br/>", styles["Normal"]))

    elements.append(Paragraph(
        "This report is generated by the Snakebite ASV Severity Predictor. "
        "It is intended only as a clinical decision support tool.",
        styles["Italic"]
    ))

    doc.build(elements)

    return pdf_file

def predictor_page():
    st.title("Snakebite ASV Severity Predictor")
    st.caption("AI-Based Clinical Decision Support System")

    col1, col2 = st.columns([5, 1])

    with col1:
        st.write(f"👋 Welcome, **{st.session_state.user_name}**")

    with col2:
        if st.button("🚪 Logout"):
            st.session_state.logged_in = False
            st.session_state.user_email = ""
            st.session_state.user_name = ""
            st.session_state.page = "login"
            st.rerun()

    if st.button("📜 View Prediction History"):
        st.session_state.page = "history"
        st.rerun()

    # Rest of your prediction code starts here...

    st.write("Enter patient details to predict ASV severity category")

    st.header("Patient Information")

    patient_name = st.text_input("Patient Name (Optional)")

    gender = st.selectbox("Gender", ["Male", "Female"])

    age_cat = st.selectbox("Age Group", [
        '1-10 years', '11-20 years', '21-30 years', '31-40 years',
        '41-50 years', '51-60 years', '>60 years'
    ])

    snake = st.selectbox("Snake Identified", [
        'Not Identified', 'Cobra', 'Krait', 'Viper', 'Python',
        'Rat Snake', 'Wolf Snake', 'Trinket', 'Humped Nose Viper',
        'Green Vine Snake', 'Sand Boa', 'Cat Snake'
    ])

    wbct = st.selectbox("20 Minute WBCT Result", ["Clotted", "Not Clotted"])

    local_ss = st.selectbox("Local Signs and Symptoms", ["Yes", "No"])

    systemic_ss = st.selectbox("Systemic Signs and Symptoms", ["Yes", "No"])

    st.header("Lab Values")

    hb = st.number_input("Hemoglobin (Hb)", value=12.0, step=0.1)
    platelets = st.number_input("Platelets", value=250.0, step=1.0)
    pt = st.number_input("PT (Prothrombin Time)", value=13.0, step=0.1)
    inr = st.number_input("INR", value=1.0, step=0.1)
    aptt = st.number_input("aPTT", value=30.0, step=0.1)
    urea = st.number_input("Urea", value=25.0, step=1.0)
    creatinine = st.number_input("Creatinine", value=1.0, step=0.1)
    sgot = st.number_input("SGOT", value=30.0, step=1.0)
    sgpt = st.number_input("SGPT", value=30.0, step=1.0)
    crp = st.number_input("CRP", value=5.0, step=0.1)

    if st.button("Predict ASV Severity"):

        input_dict = {
            'Hb': hb, 'Platelets': platelets, 'PT': pt, 'INR': inr,
            'aPTT': aptt, 'Urea': urea, 'Creatinine': creatinine,
            'SGOT': sgot, 'SGPT': sgpt, 'CRP': crp,
            'Gender(1,2)': gender,
            'Age (Cat)': age_cat,
            'Snake identified': snake,
            '20WBCT': wbct,
            'Local S/S': local_ss,
            'Systemic s/s': systemic_ss
        }

    

        input_df = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)

        final_input = pd.DataFrame(columns=feature_list)
        final_input = pd.concat([final_input, input_encoded], ignore_index=True)
        final_input = final_input.reindex(columns=feature_list, fill_value=0)
        final_input = final_input.fillna(0)

        # Convert all columns to numeric
        final_input = final_input.apply(pd.to_numeric, errors='coerce').fillna(0)

        # Get prediction
        prediction = model.predict(final_input)[0]

        # Get probabilities for each category
        probabilities = model.predict_proba(final_input)[0]

        severity_map = {0: 'None', 1: 'Low', 2: 'Moderate', 3: 'High'}
        result = severity_map[prediction]

        vial_ranges = {
            'None'     : '0 vials',
            'Low'      : '10–15 vials',
            'Moderate' : '20–30 vials',
            'High'     : '35–55 vials'
        }

        st.header("Prediction Result")

        if result == 'None':
            st.success(f"Predicted Severity: **{result}**")
            st.info("💉 Recommended ASV: **0 vials** — No envenomation, observation only")
        elif result == 'Low':
            st.info(f"Predicted Severity: **{result}** — Mild envenomation")
            st.warning(f"💉 Recommended ASV: **{vial_ranges[result]}** — Monitor closely after each dose")
        elif result == 'Moderate':
            st.warning(f"Predicted Severity: **{result}** — Moderate envenomation")
            st.warning(f"💉 Recommended ASV: **{vial_ranges[result]}** — Reassess after every 10 vials")
        else:
            st.error(f"Predicted Severity: **{result}** — Severe envenomation!")
            st.error(f"💉 Recommended ASV: **{vial_ranges[result]}** — ICU monitoring required, reassess frequently")

        # Show probability breakdown
        st.header("Model Confidence")
        st.write("How confident is the model about each severity level?")

        prob_df = pd.DataFrame({
            'Severity' : ['None', 'Low', 'Moderate', 'High'],
            'Confidence (%)' : [round(float(p) * 100, 1) for p in probabilities]
        })

        for _, row in prob_df.iterrows():
            st.progress(
                int(row['Confidence (%)']),
                text=f"{row['Severity']} : {row['Confidence (%)']}%"
            )

        # Save prediction to Firestore
        db.collection("predictions").add({
            "user_email": st.session_state.user_email,
            "user_name": st.session_state.user_name,
            "gender": gender,
            "age_group": age_cat,
            "snake": snake,
            "severity": result,
            "recommended_asv": vial_ranges[result],
            "timestamp": firestore.SERVER_TIMESTAMP
        })

        pdf_path = create_pdf(
        patient_name,
        gender,
        age_cat,
        snake,
        hb,
        platelets,
        pt,
        inr,
        aptt,
        urea,
        creatinine,
        sgot,
        sgpt,
        crp,
        result,
        vial_ranges[result]
        )

        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Prediction Report",
                data=pdf_file,
                file_name="Snakebite_ASV_Report.pdf",
                mime="application/pdf"
            )

        st.caption("⚠️ This is a decision support tool only. Final dosage must be determined by a qualified physician.")

def history_page():

    st.title("Prediction History")

    if st.button("⬅ Back to Prediction"):
        st.session_state.page = "predict"
        st.rerun()

    st.write(f"Logged in as: **{st.session_state.user_name}**")

    # Get all predictions of the logged-in user
    docs = db.collection("predictions") \
            .where("user_email", "==", st.session_state.user_email) \
            .stream()

    found = False

    for doc in docs:

        found = True

        data = doc.to_dict()

        st.subheader(f"🐍 {data['snake']}")

        st.write(f"**Severity:** {data['severity']}")
        st.write(f"**Recommended ASV:** {data['recommended_asv']}")

        if data.get("timestamp"):
            st.write(f"**Date:** {data['timestamp']}")

        st.divider()

    if not found:
        st.info("No prediction history found.")

if st.session_state.page == "login":
    login_page()

elif st.session_state.page == "register":
    register_page()

elif st.session_state.page == "predict":
    if st.session_state.logged_in:
        predictor_page()
    else:
        st.session_state.page = "login"
        st.rerun()

elif st.session_state.page == "history":

    if st.session_state.logged_in:
        history_page()
    else:
        st.session_state.page = "login"
        st.rerun()
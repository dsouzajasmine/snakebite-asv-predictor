import streamlit as st
import pandas as pd
import joblib

model = joblib.load('asv_model.pkl')
feature_list = joblib.load('model_features.pkl')

st.title("🐍 Snakebite ASV Severity Predictor")
st.write("Enter patient details to predict ASV severity category")

st.header("Patient Information")

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

    st.caption("⚠️ This is a decision support tool only. Final dosage must be determined by a qualified physician.")

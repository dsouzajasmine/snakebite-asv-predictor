import streamlit as st
import pandas as pd
import joblib
import requests
import firebase_admin
from firebase_admin import credentials, firestore, auth
from reportlab.platypus import SimpleDocTemplate, Paragraph
from reportlab.lib.styles import getSampleStyleSheet
from datetime import datetime
import plotly.express as px

# ── Page Config ──────────────────────────────────────
st.set_page_config(
    page_icon="logo.png",
    page_title = "Snakebite ASV Predictor",
    layout     = "wide"
)

st.markdown("""
<style>

/* Sidebar */
[data-testid="stSidebar"]{
    background:#0B3D2E;
}

[data-testid="stSidebar"] *{
    color:white;
}

/* Navigation buttons */
[data-testid="stSidebar"] .stButton>button{
    width:100%;
    border-radius:12px;
    background:transparent;
    border:1px solid rgba(255,255,255,0.15);
    color:white;
    font-weight:600;
    transition:0.3s;
}

[data-testid="stSidebar"] .stButton>button:hover{
    background:#1B5E49;
    border-color:#1B5E49;
    color:white;
}

/* Divider */
[data-testid="stSidebar"] hr{
    border-color:rgba(255,255,255,.18);
}

</style>
""", unsafe_allow_html=True)

# ── Session State ────────────────────────────────────
if "page"       not in st.session_state: st.session_state.page       = "login"
if "logged_in"  not in st.session_state: st.session_state.logged_in  = False
if "user_email" not in st.session_state: st.session_state.user_email = ""
if "user_name"  not in st.session_state: st.session_state.user_name  = ""
if "user_id"    not in st.session_state: st.session_state.user_id    = ""
if "role"       not in st.session_state: st.session_state.role       = ""
if "registered" not in st.session_state: st.session_state.registered = False

# ── Load Model ───────────────────────────────────────
model        = joblib.load('asv_model.pkl')
feature_list = joblib.load('model_features.pkl')

# ── Firebase Init ────────────────────────────────────
if not firebase_admin._apps:
    try:
        firebase_cred = dict(st.secrets["firebase_key"])
        cred = credentials.Certificate(firebase_cred)
    except:
        cred = credentials.Certificate('firebase_key.json')
    firebase_admin.initialize_app(cred)

db = firestore.client()

# ── Helper: Get API Key ──────────────────────────────
def get_api_key():
    try:
        return st.secrets["firebase_key"]["api_key"]
    except:
        return ""

# ── Helper: Login User ───────────────────────────────
def login_user(email, password):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:signInWithPassword?key={get_api_key()}"
    payload = {"email": email, "password": password, "returnSecureToken": True}
    return requests.post(url, json=payload)

# ── Helper: Get User Role ─────────────────────────────
def get_user_role(uid):
    try:
        doc = db.collection("users").document(uid).get()
        if doc.exists:
            return doc.to_dict().get("role", "doctor")
        return "doctor"
    except:
        return "doctor"

# ── Helper: Get User Status ───────────────────────────
def get_user_status(uid):
    try:
        doc = db.collection("users").document(uid).get()
        if doc.exists:
            return doc.to_dict().get("status", "pending")
        return "pending"
    except:
        return "pending"

# ── Helper: Forgot Password ───────────────────────────
def send_password_reset(email):
    url = f"https://identitytoolkit.googleapis.com/v1/accounts:sendOobCode?key={get_api_key()}"
    payload = {"requestType": "PASSWORD_RESET", "email": email}
    return requests.post(url, json=payload)

# ── Helper: Create PDF ───────────────────────────────
def create_pdf(patient_name, gender, age_group, snake,
               hb, platelets, pt, inr, aptt, urea,
               creatinine, sgot, sgpt, crp, severity, asv):
    pdf_file = "Snakebite_Report.pdf"
    doc      = SimpleDocTemplate(pdf_file)
    styles   = getSampleStyleSheet()
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
    elements.append(Paragraph(f"<b>Hb:</b> {hb}", styles["Normal"]))
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
    elements.append(Paragraph(f"<b>Generated On:</b> {datetime.now().strftime('%d %B %Y %I:%M %p')}", styles["Normal"]))
    elements.append(Paragraph("<br/>", styles["Normal"]))
    elements.append(Paragraph("This report is generated by the Snakebite ASV Severity Predictor. It is intended only as a clinical decision support tool.", styles["Italic"]))
    doc.build(elements)
    return pdf_file

# ── Sidebar - Doctor ──────────────────────────────────────────
def show_doctor_sidebar():

    with st.sidebar:

        col1, col2 = st.columns([1,3])

        with col1:
            st.image("logo.png", width=60)

        with col2:

            st.markdown("""
            <div style="padding-top:5px;">
                <div style="
                    font-size:22px;
                    font-weight:700;
                    color:white;
                    line-height:1;">
                    Snakebite ASV Predictor
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.divider()

        st.markdown(f"""
        <div style="font-size:17px;font-weight:600;">
        👤 {st.session_state.user_name}
        </div>

        <div style="font-size:14px;color:#d9d9d9;">
        📧 {st.session_state.user_email}
        </div>
        """, unsafe_allow_html=True)

        st.divider()

        st.markdown("### Navigation")

        if st.button("🔍 Predict", use_container_width=True):
            st.session_state.page="predict"
            st.rerun()

        if st.button("📊 Dashboard", use_container_width=True):
            st.session_state.page="dashboard"
            st.rerun()

        if st.button("📜 Prediction History", use_container_width=True):
            st.session_state.page="history"
            st.rerun()

        st.divider()

        if st.button("🚪 Logout", use_container_width=True):
            for key in ["logged_in", "user_email", "user_name", "user_id", "role"]:
                st.session_state[key] = "" if key != "logged_in" else False
            st.session_state.page = "login"
            st.rerun()

# ── Sidebar — Admin ───────────────────────────────────
def show_admin_sidebar():
    with st.sidebar:
        col1, col2 = st.columns([1, 3])
        with col1:
            st.image("logo.png", width=60)
        with col2:
            st.markdown("""
            <div style="padding-top:5px;">
                <div style="font-size:18px;font-weight:700;color:white;line-height:1;">
                    Snakebite ASV Predictor
                </div>
            </div>
            """, unsafe_allow_html=True)
        st.divider()
        st.markdown(f"""
        <div style="font-size:17px;font-weight:600;">👤 {st.session_state.user_name or "Admin"}</div>
        <div style="font-size:14px;color:#d9d9d9;">📧 {st.session_state.user_email}</div>
        <div style="font-size:12px;color:#FFD700;margin-top:4px;">⭐ Admin</div>
        """, unsafe_allow_html=True)
        st.divider()
        st.markdown("### Admin Panel")
        if st.button("📋 Manage Doctors", use_container_width=True):
            st.session_state.page = "admin_doctors"
            st.rerun()
        if st.button("📊 Hospital Dashboard", use_container_width=True):
            st.session_state.page = "admin_dashboard"
            st.rerun()
        if st.button("📜 All Predictions", use_container_width=True):
            st.session_state.page = "admin_predictions"
            st.rerun()
        st.divider()
        if st.button("🚪 Logout", use_container_width=True):
            for key in ["logged_in", "user_email", "user_name", "user_id", "role"]:
                st.session_state[key] = "" if key != "logged_in" else False
            st.session_state.page = "login"
            st.rerun()

# ── Login Page ───────────────────────────────────────
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Snakebite ASV Predictor")
        st.caption("AI-Based Clinical Decision Support System")
        st.divider()
        if st.session_state.registered:
            st.success("✅ Registration successful! Please login with your credentials.")
            st.session_state.registered = False
        st.subheader("Login")
        email    = st.text_input("Email")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            if not email or not password:
                st.error("Please enter email and password!")
            else:
                response = login_user(email, password)
                if response.status_code == 200:
                    user   = auth.get_user_by_email(email)
                    role   = get_user_role(user.uid)
                    status = get_user_status(user.uid)

                    if role == "doctor" and status == "pending":
                        st.warning("⏳ Your account is pending admin approval. Please wait.")
                    elif role == "doctor" and status == "rejected":
                        st.error("❌ Your account has been rejected. Please contact the hospital admin.")
                    else:
                        st.session_state.logged_in  = True
                        st.session_state.user_email = email
                        st.session_state.user_name  = user.display_name
                        st.session_state.user_id    = user.uid
                        st.session_state.role       = role
                        st.session_state.page       = "predict" if role == "doctor" else "admin_doctors"
                        st.success(f"Welcome, {user.display_name}!")
                        st.rerun()
                else:
                    st.error("Invalid Email or Password!")

        col_a, col_b = st.columns(2)
        with col_a:
            if st.button("Register here", use_container_width=True):
                st.session_state.page = "register"
                st.rerun()
        with col_b:
            if st.button("Forgot Password?", use_container_width=True):
                st.session_state.page = "forgot_password"
                st.rerun()

# ── Forgot Password Page ──────────────────────────────
def forgot_password_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Snakebite ASV Predictor")
        st.caption("AI-Based Clinical Decision Support System")
        st.divider()
        st.subheader("Forgot Password")
        st.write("Enter your registered email address. We will send you a password reset link.")
        email = st.text_input("Registered Email")

        if st.button("Send Reset Link", use_container_width=True):
            if not email:
                st.error("Please enter your email!")
            else:
                response = send_password_reset(email)
                if response.status_code == 200:
                    st.success("✅ Password reset link sent! Please check your email inbox.")
                else:
                    st.error("Email not found! Please check the email address and try again.")

        if st.button("Back to Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()


# ── Register Page ────────────────────────────────────
def register_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.title("Snakebite ASV Predictor")
        st.caption("AI-Based Clinical Decision Support System")
        st.divider()
        st.subheader("Doctor Registration")
        name             = st.text_input("Full Name")
        email            = st.text_input("Email")
        password         = st.text_input("Password", type="password")
        confirm_password = st.text_input("Confirm Password", type="password")

        if st.button("Register", use_container_width=True):
            if not name or not email or not password or not confirm_password:
                st.error("Please fill all fields!")
            elif password != confirm_password:
                st.error("Passwords do not match!")
            elif len(password) < 6:
                st.error("Password must be at least 6 characters!")
            else:
                try:
                    user = auth.create_user(
                        email        = email,
                        password     = password,
                        display_name = name
                    )
                    db.collection("users").document(user.uid).set({
                        "name" : name,
                        "email": email,
                        "role": "doctor",
                        "status": "pending",
                        "registered_on": firestore.SERVER_TIMESTAMP
                    })
                    st.session_state.registered = True
                    st.session_state.page = "login"
                    st.rerun()
                except Exception as e:
                    st.error(f"Registration failed: {e}")

        if st.button("Back to Login", use_container_width=True):
            st.session_state.page = "login"
            st.rerun()

# ── Admin: Manage Doctors Page ────────────────────────
def admin_doctors_page():
    show_admin_sidebar()
    st.title("📋 Manage Doctors")
    st.caption("Approve or reject doctor registrations")
    st.divider()

    # Get all doctors
    docs = db.collection("users").where("role", "==", "doctor").stream()
    doctors = [{"id": doc.id, **doc.to_dict()} for doc in docs]

    pending  = [d for d in doctors if d.get("status") == "pending"]
    approved = [d for d in doctors if d.get("status") == "approved"]
    rejected = [d for d in doctors if d.get("status") == "rejected"]

    # Metrics
    col1, col2, col3 = st.columns(3)
    col1.metric("⏳ Pending Approval", len(pending))
    col2.metric("✅ Approved Doctors", len(approved))
    col3.metric("❌ Rejected", len(rejected))
    st.divider()

    # Pending doctors
    st.subheader("⏳ Pending Approval")
    if not pending:
        st.info("No pending registrations!")
    else:
        for doc in pending:
            with st.expander(f"👤 {doc.get('name', 'N/A')} — {doc.get('email', 'N/A')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {doc.get('name', 'N/A')}")
                    st.write(f"**Email:** {doc.get('email', 'N/A')}")
                    if doc.get('registered_on'):
                        st.write(f"**Registered On:** {doc.get('registered_on')}")
                with col2:
                    if st.button(f"✅ Approve", key=f"approve_{doc['id']}"):
                        db.collection("users").document(doc['id']).update({"status": "approved"})
                        st.success(f"✅ {doc.get('name')} approved!")
                        st.rerun()
                    if st.button(f"❌ Reject", key=f"reject_{doc['id']}"):
                        db.collection("users").document(doc['id']).update({"status": "rejected"})
                        st.error(f"❌ {doc.get('name')} rejected!")
                        st.rerun()

    st.divider()

    # Approved doctors
    st.subheader("✅ Approved Doctors")
    if not approved:
        st.info("No approved doctors yet!")
    else:
        for doc in approved:
            with st.expander(f"👤 {doc.get('name', 'N/A')} — {doc.get('email', 'N/A')}"):
                col1, col2 = st.columns(2)
                with col1:
                    st.write(f"**Name:** {doc.get('name', 'N/A')}")
                    st.write(f"**Email:** {doc.get('email', 'N/A')}")
                    if doc.get('registered_on'):
                        st.write(f"**Registered On:** {doc.get('registered_on')}")
                with col2:
                    if st.button(f"🚫 Revoke Access", key=f"revoke_{doc['id']}"):
                        db.collection("users").document(doc['id']).update({"status": "rejected"})
                        st.warning(f"Access revoked for {doc.get('name')}!")
                        st.rerun()

# ── Admin: Hospital Dashboard ─────────────────────────
def admin_dashboard_page():
    show_admin_sidebar()
    st.title("📊 Hospital Dashboard")
    st.caption("Overall hospital statistics and analytics")
    st.divider()

    # Get all predictions
    all_preds = [doc.to_dict() for doc in db.collection("predictions").stream()]
    all_doctors = [doc.to_dict() for doc in db.collection("users").where("role", "==", "doctor").stream()]

    if not all_preds:
        st.info("No predictions made yet!")
        return

    df_all = pd.DataFrame(all_preds)

    # Metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🏥 Total Predictions", len(df_all))
    col2.metric("👨‍⚕️ Total Doctors", len(all_doctors))
    col3.metric("🚨 High Severity Cases", len(df_all[df_all['severity'] == 'High']))
    col4.metric("✅ No ASV Needed", len(df_all[df_all['severity'] == 'None']))
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        severity_counts = df_all['severity'].value_counts().reset_index()
        severity_counts.columns = ['Severity', 'Count']
        fig = px.pie(
            severity_counts,
            names='Severity', values='Count',
            title='Hospital-wide Severity Distribution',
            color_discrete_map={
                'None': '#4CAF50', 'Low': '#8BC34A',
                'Moderate': '#FFC107', 'High': '#F44336'
            }
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        snake_counts = df_all['snake'].value_counts().reset_index()
        snake_counts.columns = ['Snake', 'Count']
        fig2 = px.bar(
            snake_counts, x='Snake', y='Count',
            title='Snake Types — All Cases',
            color='Count', color_continuous_scale='Greens'
        )
        st.plotly_chart(fig2, use_container_width=True)

# ── Admin: All Predictions Page ───────────────────────
def admin_predictions_page():
    show_admin_sidebar()
    st.title("📜 All Predictions")
    st.caption("All predictions made by all doctors")
    st.divider()

    docs    = db.collection("predictions").stream()
    records = [doc.to_dict() for doc in docs]
    records.reverse()

    if not records:
        st.info("No predictions found!")
        return

    st.write(f"**Total Predictions: {len(records)}**")
    st.divider()

    for i, data in enumerate(records, 1):
        with st.expander(f"Prediction {i} — Dr. {data.get('user_name', 'N/A')} — {data.get('snake', 'N/A')} — Severity: {data.get('severity', 'N/A')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Doctor:** {data.get('user_name', 'N/A')}")
                st.write(f"**Doctor Email:** {data.get('user_email', 'N/A')}")
                st.write(f"**Patient:** {data.get('patient_name', 'Not provided')}")
                st.write(f"**Gender:** {data.get('gender', 'N/A')}")
                st.write(f"**Age Group:** {data.get('age_group', 'N/A')}")
                st.write(f"**Snake:** {data.get('snake', 'N/A')}")
            with col2:
                st.write(f"**Severity:** {data.get('severity', 'N/A')}")
                st.write(f"**Recommended ASV:** {data.get('recommended_asv', 'N/A')}")
                st.write(f"**WBCT:** {data.get('wbct', 'N/A')}")
                if data.get('timestamp'):
                    st.write(f"**Date:** {data.get('timestamp')}")

# ── Doctor: Predict Page ──────────────────────────────
def predict_page():
    show_doctor_sidebar()
    st.title("🔍 ASV Severity Prediction")
    st.caption("Enter patient details to predict ASV severity")
    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Patient Information")
        patient_name = st.text_input("Patient Name (Optional)")
        gender       = st.selectbox("Gender", ["Male", "Female"])
        age_cat      = st.selectbox("Age Group", [
            '1-10 years', '11-20 years', '21-30 years', '31-40 years',
            '41-50 years', '51-60 years', '>60 years'
        ])
        snake = st.selectbox("Snake Identified", [
            'Not Identified', 'Cobra', 'Krait', 'Viper', 'Python',
            'Rat Snake', 'Wolf Snake', 'Trinket', 'Humped Nose Viper',
            'Green Vine Snake', 'Sand Boa', 'Cat Snake'
        ])
        wbct        = st.selectbox("20 Minute WBCT Result", ["Clotted", "Not Clotted"])
        local_ss    = st.selectbox("Local Signs and Symptoms", ["Yes", "No"])
        systemic_ss = st.selectbox("Systemic Signs and Symptoms", ["Yes", "No"])

    with col2:
        st.subheader("Lab Values")
        hb         = st.number_input("Hemoglobin (Hb)", value=12.0, step=0.1)
        platelets  = st.number_input("Platelets", value=250.0, step=1.0)
        pt         = st.number_input("PT", value=13.0, step=0.1)
        inr        = st.number_input("INR", value=1.0, step=0.1)
        aptt       = st.number_input("aPTT", value=30.0, step=0.1)
        urea       = st.number_input("Urea", value=25.0, step=1.0)
        creatinine = st.number_input("Creatinine", value=1.0, step=0.1)
        sgot       = st.number_input("SGOT", value=30.0, step=1.0)
        sgpt       = st.number_input("SGPT", value=30.0, step=1.0)
        crp        = st.number_input("CRP", value=5.0, step=0.1)

    st.divider()

    if st.button("🔍 Predict ASV Severity", use_container_width=True):
        input_dict = {
            'Hb': hb, 'Platelets': platelets, 'PT': pt, 'INR': inr,
            'aPTT': aptt, 'Urea': urea, 'Creatinine': creatinine,
            'SGOT': sgot, 'SGPT': sgpt, 'CRP': crp,
            'Gender(1,2)': gender, 'Age (Cat)': age_cat,
            'Snake identified': snake, '20WBCT': wbct,
            'Local S/S': local_ss, 'Systemic s/s': systemic_ss
        }

        input_df      = pd.DataFrame([input_dict])
        input_encoded = pd.get_dummies(input_df)
        final_input   = pd.DataFrame(columns=feature_list)
        final_input   = pd.concat([final_input, input_encoded], ignore_index=True)
        final_input   = final_input.reindex(columns=feature_list, fill_value=0)
        final_input   = final_input.fillna(0)
        final_input   = final_input.apply(pd.to_numeric, errors='coerce').fillna(0)

        prediction    = model.predict(final_input)[0]
        probabilities = model.predict_proba(final_input)[0]
        severity_map  = {0: 'None', 1: 'Low', 2: 'Moderate', 3: 'High'}
        result        = severity_map[prediction]
        vial_ranges   = {
            'None'    : '0 vials',
            'Low'     : '10–15 vials',
            'Moderate': '20–30 vials',
            'High'    : '35–55 vials'
        }

        st.divider()
        st.header("Prediction Result")

        col1, col2 = st.columns(2)
        with col1:
            if result == 'None':
                st.success(f"### Predicted Severity: {result}")
                st.info("💉 Recommended ASV: **0 vials**\nNo envenomation, observation only")
            elif result == 'Low':
                st.info(f"### Predicted Severity: {result}")
                st.warning(f"💉 Recommended ASV: **{vial_ranges[result]}**\nMonitor closely after each dose")
            elif result == 'Moderate':
                st.warning(f"### Predicted Severity: {result}")
                st.warning(f"💉 Recommended ASV: **{vial_ranges[result]}**\nReassess after every 10 vials")
            else:
                st.error(f"### Predicted Severity: {result}")
                st.error(f"💉 Recommended ASV: **{vial_ranges[result]}**\nICU monitoring required!")

        with col2:
            st.subheader("Model Confidence")
            prob_df = pd.DataFrame({
                'Severity'      : ['None', 'Low', 'Moderate', 'High'],
                'Confidence (%)': [round(float(p) * 100, 1) for p in probabilities]
            })
            for _, row in prob_df.iterrows():
                st.progress(
                    int(row['Confidence (%)']),
                    text=f"{row['Severity']} : {row['Confidence (%)']:.1f}%"
                )

        # Save to Firestore
        db.collection("predictions").add({
            "user_email"     : st.session_state.user_email,
            "user_name"      : st.session_state.user_name,
            "patient_name"   : patient_name,
            "gender"         : gender,
            "age_group"      : age_cat,
            "snake"          : snake,
            "wbct"           : wbct,
            "local_ss"       : local_ss,
            "systemic_ss"    : systemic_ss,
            "severity"       : result,
            "recommended_asv": vial_ranges[result],
            "timestamp"      : firestore.SERVER_TIMESTAMP
        })

        st.success("✅ Prediction saved to history!")

        pdf_path = create_pdf(
            patient_name, gender, age_cat, snake,
            hb, platelets, pt, inr, aptt, urea,
            creatinine, sgot, sgpt, crp,
            result, vial_ranges[result]
        )
        with open(pdf_path, "rb") as pdf_file:
            st.download_button(
                label="📄 Download Prediction Report",
                data=pdf_file,
                file_name="Snakebite_ASV_Report.pdf",
                mime="application/pdf"
            )
        st.caption("⚠️ This is a decision support tool only. Final dosage must be determined by a qualified physician.")

# ── Doctor - Dashboard Page ───────────────────────────────────
def dashboard_page():
    show_doctor_sidebar()

    st.title("📊 Dashboard")
    st.markdown(
        f"""
        ### Welcome back, {st.session_state.user_name}!
        Monitor your prediction history, severity trends and patient statistics.
        """
    )
    st.divider()

    docs = db.collection("predictions")\
             .where("user_email", "==", st.session_state.user_email)\
             .stream()

    records = [doc.to_dict() for doc in docs]

    if not records:
        st.info("No predictions yet! Make a prediction first.")
        return

    df_hist = pd.DataFrame(records)

    # -------------------- Metrics --------------------
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("📋 Total Predictions", len(df_hist))
    col2.metric("🚨 High Severity",
                len(df_hist[df_hist["severity"] == "High"]))
    col3.metric("⚠ Moderate Severity",
                len(df_hist[df_hist["severity"] == "Moderate"]))
    col4.metric("✅ No ASV Needed",
                len(df_hist[df_hist["severity"] == "None"]))

    st.divider()

    # -------------------- Row 1 --------------------

    col1, col2 = st.columns(2)

    with col1:

        severity_counts = df_hist["severity"].value_counts().reset_index()
        severity_counts.columns = ["Severity", "Count"]

        fig = px.pie(
            severity_counts,
            names="Severity",
            values="Count",
            title="Severity Distribution",
            color_discrete_map={
                "None": "#4CAF50",
                "Low": "#8BC34A",
                "Moderate": "#FFC107",
                "High": "#F44336"
            }
        )

        fig.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=15)
        )

        st.plotly_chart(fig, use_container_width=True)

        st.caption(
            "Shows the proportion of predictions belonging to each severity category."
        )

    with col2:

        snake_counts = df_hist["snake"].value_counts().reset_index()
        snake_counts.columns = ["Snake", "Count"]

        fig2 = px.bar(
            snake_counts,
            x="Snake",
            y="Count",
            title="Snake Identification Frequency",
            color="Count",
            color_continuous_scale="Tealgrn"
        )

        fig2.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=15)
        )

        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Displays how frequently each snake species has appeared in previous predictions."
        )

    st.divider()

    # -------------------- Row 2 --------------------

    col1, col2 = st.columns(2)

    with col1:

        asv_counts = df_hist["recommended_asv"].value_counts().reset_index()
        asv_counts.columns = ["ASV", "Count"]

        fig3 = px.bar(
        asv_counts,
        x="ASV",
        y="Count",
        title="Recommended ASV Distribution",
        color="ASV",
        color_discrete_map={
            "0 vials": "#4CAF50",
            "10–15 vials": "#8BC34A",
            "20–30 vials": "#FFC107",
            "35–55 vials": "#F44336"
        }
        )
    
        fig3.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=15)
        )

        st.plotly_chart(fig3, use_container_width=True)

        st.caption(
            "Shows the frequency of each recommended ASV dosage range."
        )

    with col2:

        gender_counts = df_hist["gender"].value_counts().reset_index()
        gender_counts.columns = ["Gender", "Count"]

        fig4 = px.pie(
            gender_counts,
            names="Gender",
            values="Count",
            title="Gender Distribution",
            color_discrete_sequence=px.colors.qualitative.Set2
        )

        fig4.update_layout(
            paper_bgcolor="white",
            plot_bgcolor="white",
            font=dict(size=15)
        )

        st.plotly_chart(fig4, use_container_width=True)

        st.caption(
            "Represents the gender distribution of patients in the prediction history."
        )

    st.divider()

    # -------------------- Recent Activity --------------------

    st.subheader("📝 Recent Prediction Activity")

    display_df = df_hist[
        [
            "patient_name",
            "snake",
            "severity",
            "recommended_asv"
        ]
    ].copy()

    display_df.columns = [
        "Patient",
        "Snake",
        "Severity",
        "Recommended ASV"
    ]

    st.dataframe(
        display_df.tail(5).iloc[::-1],
        use_container_width=True,
        hide_index=True
    )

# ── Doctoe - History Page ─────────────────────────────────────
def history_page():
    show_doctor_sidebar()
    st.title("📜 Prediction History")
    st.markdown("View all your previous snakebite severity predictions.")
    st.divider()

    docs = db.collection("predictions")\
             .where("user_email", "==", st.session_state.user_email)\
             .stream()

    records = [doc.to_dict() for doc in docs]
    records.reverse()

    if not records:
        st.info("No prediction history found.")
        return

    for i, data in enumerate(records, 1):
        with st.expander(f"Prediction {i} — {data.get('snake', 'N/A')} — Severity: {data.get('severity', 'N/A')}"):
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"**Patient:** {data.get('patient_name', 'Not provided')}")
                st.write(f"**Gender:** {data.get('gender', 'N/A')}")
                st.write(f"**Age Group:** {data.get('age_group', 'N/A')}")
                st.write(f"**Snake:** {data.get('snake', 'N/A')}")
                st.write(f"**WBCT:** {data.get('wbct', 'N/A')}")
            with col2:
                st.write(f"**Severity:** {data.get('severity', 'N/A')}")
                st.write(f"**Recommended ASV:** {data.get('recommended_asv', 'N/A')}")
                if data.get('timestamp'):
                    st.write(f"**Date:** {data.get('timestamp')}")

# ── Router ────────────────────────────────────────────
if not st.session_state.logged_in:
    if st.session_state.page == "register":
        register_page()
    elif st.session_state.page == "forgot_password":
        forgot_password_page()
    else:
        login_page()
else:
    role = st.session_state.role

    if role == "admin":
        if st.session_state.page == "admin_doctors":
            admin_doctors_page()
        elif st.session_state.page == "admin_dashboard":
            admin_dashboard_page()
        elif st.session_state.page == "admin_predictions":
            admin_predictions_page()
        else:
            admin_doctors_page()
    else:
        if st.session_state.page == "predict":
            predict_page()
        elif st.session_state.page == "dashboard":
            dashboard_page()
        elif st.session_state.page == "history":
            history_page()
        else:
            predict_page()
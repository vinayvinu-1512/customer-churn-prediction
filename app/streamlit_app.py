# app/streamlit_app.py
import os
import io
import base64
import requests
from datetime import datetime

import pandas as pd
import joblib
import streamlit as st
import plotly.express as px
from fpdf import FPDF
from streamlit_lottie import st_lottie

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Customer Churn SaaS Demo", layout="wide", page_icon="📊")

# ------------------ DARK NEON CSS (STYLE A) ------------------
DARK_NEON_CSS = """
<style>
/* Page background & fonts */
.stApp {
  background: radial-gradient(circle at 10% 10%, #071226 0%, #05060a 40%, #020205 100%);
  color: #cfe8ff;
  font-family: "Segoe UI", Roboto, "Helvetica Neue", Arial;
  -webkit-font-smoothing: antialiased;
}

/* Card style */
.card {
  background: linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.02));
  border: 1px solid rgba(255,255,255,0.04);
  padding: 18px;
  border-radius: 12px;
  box-shadow: 0 8px 30px rgba(16, 24, 40, 0.5);
}

/* Title */
.main-title {
  font-size: 34px;
  font-weight: 800;
  text-align: center;
  background: linear-gradient(90deg,#3dd1ff,#6d5cff,#a06bff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  margin-bottom: 6px;
}

/* Button style */
.stButton>button {
  background: linear-gradient(90deg,#0ea5e9,#7c3aed);
  color: white;
  border-radius: 10px;
  padding: 10px 20px;
  font-weight: 700;
  border: none;
}
.stButton>button:hover {
  transform: translateY(-2px);
}

/* Result card colors */
.result-card {
  padding: 18px;
  border-radius: 12px;
  text-align: center;
  font-weight: 700;
  color: #031124;
}

/* Sidebar tweaks */
.css-1d391kg { /* small tweak for sidebar area (class may vary across versions) */ }

/* Footer */
.footer {
  text-align:center;
  opacity:0.65;
  margin-top: 18px;
  font-size: 13px;
}
</style>
"""
st.markdown(DARK_NEON_CSS, unsafe_allow_html=True)

# ------------------ MODEL LOADING ------------------
MODEL_PATH = os.environ.get("MODEL_PATH", "models/churn_model.pkl")

@st.cache_resource(show_spinner=False)
def load_model(path):
    if not os.path.exists(path):
        return None
    return joblib.load(path)

model = load_model(MODEL_PATH)

# ------------------ LOTTIE LOADER ------------------
def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except Exception:
        return None
    return None

LOTTIE_URL = "https://assets5.lottiefiles.com/packages/lf20_jcikwtux.json"
lottie_json = load_lottie(LOTTIE_URL)

# ------------------ UTILITIES ------------------
def create_pdf_report(input_dict: dict, prediction_text: str, prob: float) -> bytes:
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=14)
    pdf.cell(200, 10, txt="Customer Churn Prediction Report", ln=True, align='C')
    pdf.ln(6)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 6, txt=f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
    pdf.ln(6)
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 6, txt=f"Prediction: {prediction_text}", ln=True)
    pdf.cell(200, 6, txt=f"Probability (churn): {prob:.2f}%", ln=True)
    pdf.ln(8)
    pdf.set_font("Arial", size=11)
    pdf.cell(200, 6, txt="Input features:", ln=True)
    pdf.ln(4)
    for k, v in input_dict.items():
        pdf.cell(200, 6, txt=f" - {k}: {v}", ln=True)
    return pdf.output(dest="S").encode("latin-1")

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

def safe_get_secrets():
    """
    Returns (username, password)
    Preference order:
      1) st.secrets['credentials'] (Streamlit Cloud)
      2) Environment variables APP_USER / APP_PASS
      3) Defaults for local dev (admin/admin123)
    """
    # 1) st.secrets safely
    try:
        creds = st.secrets.get("credentials", None)
        if creds and isinstance(creds, dict):
            user = creds.get("username")
            pwd = creds.get("password")
            if user and pwd:
                return user, pwd
    except Exception:
        # If st.secrets doesn't exist or cannot be read, fall back
        pass

    # 2) env vars
    env_user = os.environ.get("APP_USER")
    env_pass = os.environ.get("APP_PASS")
    if env_user and env_pass:
        return env_user, env_pass

    # 3) local default
    return "admin", "admin123"

VALID_USER, VALID_PASS = safe_get_secrets()

# ------------------ AUTHENTICATION ------------------
def login_box():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    st.sidebar.markdown("## 🔒 Login")
    username = st.sidebar.text_input("Username")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Sign in"):
        if username == VALID_USER and password == VALID_PASS:
            st.session_state["logged_in"] = True
            st.session_state["user"] = username
            st.sidebar.success("Logged in")
        else:
            st.sidebar.error("Invalid credentials")
    return st.session_state["logged_in"]

# ------------------ BULK PREDICTION ------------------
def bulk_predict(df: pd.DataFrame) -> pd.DataFrame:
    required_cols = ['gender','SeniorCitizen','Partner','Dependents','PhoneService','MultipleLines',
                     'InternetService','OnlineSecurity','OnlineBackup','DeviceProtection','TechSupport',
                     'StreamingTV','StreamingMovies','PaperlessBilling','Contract','PaymentMethod',
                     'tenure','MonthlyCharges','TotalCharges']
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    X = df[required_cols].copy()
    preds = model.predict(X)
    probs = model.predict_proba(X)[:,1] * 100
    df_out = df.copy()
    df_out["Churn_Pred"] = preds
    df_out["Churn_Prob"] = probs
    return df_out

# ------------------ PAGES ------------------
def page_home():
    st.markdown("<div class='card'><h2 class='main-title'>📊 Customer Churn SaaS Demo</h2></div>", unsafe_allow_html=True)
    st.write("This is a polished demo of an end-to-end churn prediction application — use the sidebar to navigate.")
    if lottie_json:
        st_lottie(lottie_json, height=220)
    st.markdown("---")
    st.markdown("<div class='card'> <h4>Features</h4> <ul><li>Single customer prediction with PDF report</li><li>Bulk CSV predictions & download</li><li>Dashboard visualizations</li></ul></div>", unsafe_allow_html=True)

def page_predict():
    st.markdown("<div class='card'><h3>🔎 Single Prediction</h3></div>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        gender = st.selectbox("Gender", ['Male', 'Female'])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ['Yes', 'No'])
        Dependents = st.selectbox("Dependents", ['Yes', 'No'])
        PhoneService = st.selectbox("Phone Service", ['Yes', 'No'])
    with col2:
        MultipleLines = st.selectbox("Multiple Lines", ['Yes', 'No'])
        InternetService = st.selectbox("Internet Service", ['DSL', 'Fiber optic', 'No'])
        OnlineSecurity = st.selectbox("Online Security", ['Yes', 'No'])
        OnlineBackup = st.selectbox("Online Backup", ['Yes', 'No'])
        DeviceProtection = st.selectbox("Device Protection", ['Yes', 'No'])
    with col3:
        TechSupport = st.selectbox("Tech Support", ['Yes', 'No'])
        StreamingTV = st.selectbox("Streaming TV", ['Yes', 'No'])
        StreamingMovies = st.selectbox("Streaming Movies", ['Yes', 'No'])
        PaperlessBilling = st.selectbox("Paperless Billing", ['Yes', 'No'])
        Contract = st.selectbox("Contract", ['Month-to-month', 'One year', 'Two year'])
        PaymentMethod = st.selectbox("Payment Method", [
            'Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'
        ])
    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72)
    MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0)
    TotalCharges = st.number_input("Total Charges", min_value=0.0)

    input_dict = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'PaperlessBilling': PaperlessBilling,
        'Contract': Contract,
        'PaymentMethod': PaymentMethod,
        'tenure': tenure,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }

    if st.button("Predict"):
        if model is None:
            st.error("Model file not found. Please upload or set MODEL_PATH.")
            return
        df_input = pd.DataFrame([input_dict])
        pred = model.predict(df_input)[0]               # 1 -> churn, 0 -> stay
        churn_prob = model.predict_proba(df_input)[0][1] * 100

        if pred == 1:
            st.markdown(f"<div class='result-card' style='background:#ff7b7b;'>❌ Predicted: Customer WILL CHURN — Churn Prob: {churn_prob:.2f}%</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div class='result-card' style='background:#8ef3c5; color:#02201f;'>✔️ Predicted: Customer WILL STAY — Stay Prob: {(100 - churn_prob):.2f}%</div>", unsafe_allow_html=True)

        # PDF download
        pdf_bytes = create_pdf_report(input_dict, "Churn" if pred==1 else "Stay", churn_prob)
        st.download_button("📄 Download PDF Report", data=pdf_bytes, file_name="churn_report.pdf", mime="application/pdf")

        # Plot
        fig = px.bar(x=["Churn", "Stay"], y=[churn_prob, 100-churn_prob], labels={'x':'Outcome','y':'Probability'},
                     text=[f"{churn_prob:.1f}%", f"{100-churn_prob:.1f}%"])
        st.plotly_chart(fig, use_container_width=True)

def page_bulk():
    st.markdown("<div class='card'><h3>📥 Bulk Predictions (CSV)</h3></div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV file (must contain required columns)", type=["csv"])
    if uploaded:
        try:
            df = pd.read_csv(uploaded)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            if st.button("Run Bulk Predict"):
                if model is None:
                    st.error("Model not found.")
                    return
                try:
                    out = bulk_predict(df)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
                st.success("Bulk prediction complete — preview below")
                st.dataframe(out.head())
                csv_bytes = df_to_csv_bytes(out)
                st.download_button("📥 Download results CSV", data=csv_bytes, file_name="bulk_predictions.csv", mime="text/csv")
        except Exception as e:
            st.error(f"Could not read CSV: {e}")

def page_dashboard():
    st.markdown("<div class='card'><h3>📈 Dashboard</h3></div>", unsafe_allow_html=True)
    if os.path.exists("data/Telco-Customer-Churn.csv"):
        df = pd.read_csv("data/Telco-Customer-Churn.csv")
        if "Churn" in df.columns:
            counts = df['Churn'].value_counts(normalize=True).rename(index={"Yes":"Churn","No":"Stay"})*100
            fig = px.pie(values=counts.values, names=counts.index, title="Dataset Churn Distribution (%)")
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No dataset found in data/ for demo charts. Push Telco CSV to repo for live charts.")

# ------------------ APPLICATION ROUTER ------------------
def main():
    if "logged_in" not in st.session_state:
        st.session_state["logged_in"] = False

    logged = login_box()
    if not logged:
        st.markdown("<div class='card'><h3>Welcome — please log in from the sidebar</h3><p>Local demo credentials: <code>admin/admin123</code></p></div>", unsafe_allow_html=True)
        return

    st.sidebar.write("---")
    st.sidebar.markdown(f"Logged in as **{st.session_state.get('user', 'user')}**")
    page = st.sidebar.radio("Choose page", ["Home", "Predict single", "Bulk predict (CSV)", "Dashboard"])

    if page == "Home":
        page_home()
    elif page == "Predict single":
        page_predict()
    elif page == "Bulk predict (CSV)":
        page_bulk()
    elif page == "Dashboard":
        page_dashboard()

    st.markdown("<div class='footer'>Built by <b>Vinay K</b> — Customer Churn Prediction · Demo App</div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()

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

# ------------------ SESSION ------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "username" not in st.session_state:
    st.session_state.username = ""

# ------------------ AUTH ------------------
VALID_USER = "admin"
VALID_PASS = "admin"

def login_ui():
    st.markdown("<h2 style='text-align:center;color:#00e1ff;'>🔐 Login</h2>", unsafe_allow_html=True)

    username = st.text_input("Username", key="login_user")
    password = st.text_input("Password", type="password", key="login_pass")

    if st.button("Login"):
        if username == VALID_USER and password == VALID_PASS:
            st.session_state.logged_in = True
            st.session_state.username = username
            st.rerun()
        else:
            st.error("Invalid username or password")

# ------------------ CSS ------------------
DARK_NEON_CSS = """
<style>
/* Global UI background */
.stApp {
  background: radial-gradient(circle at 10% 10%, #071226 0%, #05060a 40%, #020205 100%);
  color: #cfe8ff;
}

/* Top-right logout button */
#logout-btn {
  position: fixed;
  right: 25px;
  top: 15px;
  background: linear-gradient(90deg,#ff4d4d,#c9184a);
  padding: 8px 16px;
  color: white;
  border-radius: 8px;
  font-weight: 700;
  cursor: pointer;
  z-index: 9999;
}

/* Title gradient */
.main-title {
  font-size: 34px;
  font-weight: 800;
  text-align: center;
  background: linear-gradient(90deg,#3dd1ff,#6d5cff,#a06bff);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
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

# ------------------ LOTTIE ------------------
def load_lottie(url: str):
    try:
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            return r.json()
    except:
        return None
    return None

LOTTIE_URL = "https://assets5.lottiefiles.com/packages/lf20_jcikwtux.json"
lottie_json = load_lottie(LOTTIE_URL)

# ------------------ PAGES ------------------
def page_home():
    col1, col2 = st.columns([1,4])
    with col1:
        st.image("app/assets/churnx_logo.png", width=120)
    with col2:
        st.markdown("<h1 class='main-title'>ChurnX — AI-Powered Churn Prediction</h1>", unsafe_allow_html=True)
        st.write("Predict, Prevent & Retain Customers with Machine Learning")

    if lottie_json:
        st_lottie(lottie_json, height=220)

def page_predict():
    st.markdown("<h3 class='main-title'>🔍 Single Prediction</h3>", unsafe_allow_html=True)
    st.write("Fill customer details below:")

def page_bulk():
    st.markdown("<h3 class='main-title'>📥 Bulk CSV Predictions</h3>", unsafe_allow_html=True)

def page_dashboard():
    st.markdown("<h3 class='main-title'>📈 Dashboard</h3>", unsafe_allow_html=True)

# ------------------ MAIN ROUTER ------------------
def main():
    # If not logged in — show pure login page (no sidebar content)
    if not st.session_state.logged_in:
        login_ui()
        return

    # SIDEBAR ONLY AFTER LOGIN
    st.sidebar.image("app/assets/churnx_logo.png", use_column_width=True)
    st.sidebar.markdown("<h2 style='text-align:center;color:white;'>ChurnX</h2>", unsafe_allow_html=True)
    st.sidebar.write("---")

    page = st.sidebar.radio("📍 Navigation", ["Home", "Predict Single", "Bulk Predict", "Dashboard"])

    # Top-right Logout Button
    st.markdown(
        """<button id="logout-btn" onclick="location.reload()">Logout</button>""",
        unsafe_allow_html=True
    )
    if st.session_state.logged_in and st.button("Logout", key="logout_hidden"):
        st.session_state.logged_in = False
        st.session_state.username = ""
        st.rerun()

    if page == "Home":
        page_home()
    elif page == "Predict Single":
        page_predict()
    elif page == "Bulk Predict":
        page_bulk()
    elif page == "Dashboard":
        page_dashboard()

if __name__ == "__main__":
    main()

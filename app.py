import base64
import json
import os

import joblib
import pandas as pd
import streamlit as st
from tensorflow import keras

st.set_page_config(page_title="AML Transaction Checker", page_icon="🛡️", layout="wide")

# ---------------------------------------------------------------- background
NETWORK_SVG = """
<svg xmlns='http://www.w3.org/2000/svg' width='1600' height='900' viewBox='0 0 1600 900'>
  <defs>
    <linearGradient id='g' x1='0' y1='0' x2='1' y2='1'>
      <stop offset='0' stop-color='#0f172a'/>
      <stop offset='0.55' stop-color='#1e1b4b'/>
      <stop offset='1' stop-color='#0c4a6e'/>
    </linearGradient>
  </defs>
  <rect width='1600' height='900' fill='url(#g)'/>
  <g stroke='#38bdf8' stroke-opacity='0.28' stroke-width='1.5' fill='none'>
    <path d='M100 180 L380 320 L690 170 L980 350 L1290 210 L1520 400'/>
    <path d='M200 700 L480 560 L760 690 L1050 540 L1340 680 L1560 560'/>
    <path d='M380 320 L480 560'/><path d='M690 170 L760 690'/>
    <path d='M980 350 L1050 540'/><path d='M1290 210 L1340 680'/>
    <path d='M100 180 L200 700'/><path d='M1520 400 L1560 560'/>
  </g>
  <g fill='#22d3ee' fill-opacity='0.7'>
    <circle cx='100' cy='180' r='7'/><circle cx='380' cy='320' r='9'/>
    <circle cx='690' cy='170' r='7'/><circle cx='980' cy='350' r='10'/>
    <circle cx='1290' cy='210' r='8'/><circle cx='1520' cy='400' r='7'/>
    <circle cx='200' cy='700' r='8'/><circle cx='480' cy='560' r='10'/>
    <circle cx='760' cy='690' r='7'/><circle cx='1050' cy='540' r='9'/>
    <circle cx='1340' cy='680' r='8'/><circle cx='1560' cy='560' r='7'/>
  </g>
  <g fill='#f472b6' fill-opacity='0.8'>
    <circle cx='980' cy='350' r='4'/><circle cx='480' cy='560' r='4'/>
  </g>
</svg>
"""


def set_background():
    if os.path.exists("background.jpg"):
        with open("background.jpg", "rb") as f:
            raw, mime = f.read(), "image/jpeg"
    else:
        raw, mime = NETWORK_SVG.encode("utf-8"), "image/svg+xml"
    b64 = base64.b64encode(raw).decode()

    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image:
                linear-gradient(rgba(8, 12, 32, 0.55), rgba(8, 12, 32, 0.55)),
                url("data:{mime};base64,{b64}");
            background-size: cover;
            background-attachment: fixed;
        }}
        [data-testid="stHeader"] {{ background: transparent; }}
        .block-container {{ padding-top: 2.2rem; max-width: 1150px; }}

        /* make every label and heading readable on the dark background */
        [data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] *,
        label, label p, h1, h2, h3, .stMarkdown, .stMarkdown p {{
            color: #f1f5f9 !important;
        }}

        .app-title {{
            font-size: 2.5rem; font-weight: 800; line-height: 1.15;
            background: linear-gradient(90deg, #22d3ee, #a78bfa, #f472b6);
            -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        }}
        .app-sub {{ color: #cbd5e1 !important; margin-bottom: 1.2rem; }}

        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.12);
            border-radius: 16px;
            backdrop-filter: blur(8px);
        }}

        .verdict {{
            padding: 1.1rem 1.3rem; border-radius: 14px;
            font-size: 1.3rem; font-weight: 700; color: white !important;
        }}
        .verdict-bad  {{ background: linear-gradient(90deg, #dc2626, #f43f5e); }}
        .verdict-good {{ background: linear-gradient(90deg, #16a34a, #22c55e); }}

        .stButton > button {{
            background: linear-gradient(90deg, #06b6d4, #8b5cf6);
            color: white; border: none; border-radius: 10px;
            font-weight: 600; padding: 0.55rem 1.4rem;
            transition: transform 0.15s ease, box-shadow 0.15s ease;
        }}
        .stButton > button:hover {{
            transform: translateY(-2px);
            box-shadow: 0 6px 18px rgba(139, 92, 246, 0.45);
            color: white;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )


set_background()


# --------------------------------------------------------------------- model
@st.cache_resource
def load_artifacts():
    preprocess = joblib.load("preprocess.pkl")
    model = keras.models.load_model("aml_model.keras")
    with open("threshold.json") as f:
        threshold = json.load(f)["threshold"]
    return preprocess, model, threshold


preprocess, model, threshold = load_artifacts()

cat_cols = list(preprocess.transformers_[0][2])
num_cols = list(preprocess.transformers_[1][2])
cat_options = dict(zip(cat_cols, preprocess.named_transformers_["cat"].categories_))


def is_suspicious(txn):
    row = pd.DataFrame([txn])[cat_cols + num_cols]
    score = float(model.predict(preprocess.transform(row), verbose=0)[0][0])
    return score >= threshold


# ---------------------------------------------------------------------- page
if "result" not in st.session_state:
    st.session_state.result = None

st.markdown('<div class="app-title">Anti-Money Laundering<br>Transaction Checker</div>',
            unsafe_allow_html=True)
st.markdown('<div class="app-sub"></div>',
            unsafe_allow_html=True)

left, right = st.columns([1.2, 1], gap="large")

with left:
    with st.container(border=True):
        st.subheader("Transaction details")
        txn = {}

        a, b = st.columns(2)
        for i, col in enumerate(cat_cols):
            target = a if i % 2 == 0 else b
            txn[col] = target.selectbox(col.replace("_", " "), cat_options[col])

        for col in num_cols:
            txn[col] = st.number_input(
                col.replace("_", " "), min_value=0.0, value=1000.0, step=100.0
            )

        if st.button("Check transaction"):
            st.session_state.result = is_suspicious(txn)

with right:
    with st.container(border=True):
        st.subheader("Result")
        result = st.session_state.result

        if result is None:
            st.write("No transaction checked yet.")
        elif result:
            st.markdown(
                '<div class="verdict verdict-bad">Suspicious Transaction</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div class="verdict verdict-good">Legitimate Transaction<br>'
                '<span class="verdict-sub">No Suspicious Activity Detected</span></div>',
                unsafe_allow_html=True,
            )

        st.write("")
        st.caption("Automated prediction. Flagged cases need manual review.")
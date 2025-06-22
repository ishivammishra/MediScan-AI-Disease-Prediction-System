import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
from utils.predict import predict_disease
from utils.helpers import get_description, get_precautions
import pickle
import time
import base64

# ---------- PAGE CONFIG ----------
st.set_page_config(
    page_title="MediScan AI",
    page_icon="🧠",
    layout="centered"
)

# ---------- LOAD CSS ----------
def load_local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_local_css("app/style.css")

# ---------- TITLE ----------
st.markdown("# 🧠 MediScan AI")
st.markdown("##### Accurate AI-powered Disease Prediction Engine")
st.markdown("<hr style='border: 1px solid #333;'>", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
with open("models/trained_model.pkl", "rb") as f:
    model, mlb, vectorizer = pickle.load(f)

all_symptoms = sorted(mlb.classes_)

# ---------- INPUT ----------
st.markdown("### 🔍 Select Your Symptoms")
selected = st.multiselect("Select one or more symptoms you are experiencing:", all_symptoms)

# ---------- PREDICTION ----------
if st.button("💡 Predict Disease"):
    if not selected:
        st.warning("⚠️ Please select at least one symptom before predicting.")
    else:
        with st.spinner("🔄 Predicting... please wait"):
            time.sleep(1.2)
            disease, confidence = predict_disease(selected)

        st.success("✅ Prediction Complete!")
        st.toast("🧠 AI has completed the diagnosis!")

        # --- Prediction Summary ---
        st.markdown("### 🎯 Prediction Result")
        st.markdown(f"""
        <div class="block">
            <h4>🧬 <strong>Predicted Disease:</strong> <code>{disease}</code></h4>
            <h4>📊 <strong>Confidence:</strong> <code>{confidence:.2f}%</code></h4>
        </div>
        """, unsafe_allow_html=True)

        # --- Description Block ---
        with st.expander("📄 View Full Disease Description"):
            st.write(get_description(disease))

        # --- Precaution Block ---
        precautions = get_precautions(disease)
        with st.expander("🩺 View Recommended Precautions"):
            if precautions and any(p.strip() for p in precautions):
                for p in precautions:
                    st.markdown(f"- ✅ {p}")
            else:
                st.info("No specific precautionary advice available for this disease.")

      

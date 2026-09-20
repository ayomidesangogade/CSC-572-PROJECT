"""
Streamlit demo app for the Phishing URL Detection project (PhiUSIIL update).

Run with:
    streamlit run app.py

Requires phishing_model_phiusiil.pkl and feature_columns_phiusiil.pkl to be
present in the same folder (produced by Section 9 of
Phishing_URL_Detection_PhiUSIIL.ipynb), plus the reference_data/ folder
(bundled alongside feature_extraction.py).
"""

import streamlit as st
import joblib
import pandas as pd
from feature_extraction2 import extract_features

st.set_page_config(page_title="Phishing URL Detector", page_icon="🔎", layout="centered")

st.title("🔎 Phishing URL Detector")
st.write(
    "Enter a URL below. The app extracts the 50 PhiUSIIL-style features used to "
    "train the model and predicts whether the URL is likely **phishing** or "
    "**legitimate**."
)

st.warning(
    "Educational coursework demo, updated for the PhiUSIIL dataset. Three "
    "features can't be computed exactly the way the original PhiUSIIL paper "
    "does, since that requires their internal reference database: "
    "**TLDLegitimateProb** is reconstructed from a per-TLD frequency table "
    "derived from the training data (verified to match closely); "
    "**URLCharProb** and **URLSimilarityIndex** are approximated using "
    "character-frequency and reference-domain lookup tables built from the "
    "same training data, and should be treated as indicative rather than "
    "exact. Predictions should not be relied on for real security decisions."
)


@st.cache_resource
def load_model():
    model = joblib.load("phishing_model_phiusiil.pkl")
    columns = joblib.load("feature_columns_phiusiil.pkl")
    return model, columns


model, feature_columns = load_model()

url = st.text_input("URL to check", placeholder="e.g. http://example.com")
check_button = st.button("Check URL", type="primary")

if check_button and url.strip():
    with st.spinner("Fetching URL and extracting features..."):
        try:
            feature_dict, extractor = extract_features(url.strip())
            vector = [feature_dict[col] for col in feature_columns]
            X = pd.DataFrame([vector], columns=feature_columns)

            prediction = model.predict(X)[0]
            proba = model.predict_proba(X)[0]

            is_phishing = prediction == 1
            confidence = proba[1] if is_phishing else proba[0]

            st.divider()
            if not extractor.fetch_succeeded():
                st.info(
                    "Couldn't fetch the page itself (site unreachable, blocked the "
                    "request, or timed out) — the prediction below is based on "
                    "URL-only features, which are less reliable on their own."
                )
            if is_phishing:
                st.error(f"⚠️ Predicted: **PHISHING** (confidence: {confidence:.1%})")
            else:
                st.success(f"✅ Predicted: **LEGITIMATE** (confidence: {confidence:.1%})")

            with st.expander("See extracted feature values"):
                st.dataframe(
                    pd.DataFrame(feature_dict.items(), columns=["Feature", "Value"]),
                    use_container_width=True,
                    hide_index=True,
                )

        except Exception as e:
            st.error(f"Could not analyze this URL: {e}")

elif check_button:
    st.info("Please enter a URL first.")

st.divider()
st.caption(
    "Model trained on the PhiUSIIL Phishing URL Dataset (235,795 labeled URLs, "
    "50 features). See the accompanying notebook for training and evaluation "
    "details."
)

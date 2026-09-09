"""
Streamlit demo app for the Phishing URL Detection project.

Run with:
    streamlit run app.py

Requires phishing_model.pkl and feature_columns.pkl to be present in the
same folder (produced by the training notebook).
"""

import streamlit as st
import joblib
import pandas as pd
from feature_extraction import extract_features

st.set_page_config(page_title="Phishing URL Detector", page_icon="🔎", layout="centered")

st.title("🔎 Phishing URL Detector")
st.write(
    "Enter a URL below. The app extracts the same 30 features used to train "
    "the model and predicts whether the URL is likely **phishing** or **legitimate**."
)

st.warning(
    "Educational coursework demo. A handful of features (web traffic rank, "
    "PageRank, Google index, backlink count, and statistical blocklist "
    "membership) rely on third-party services that are no longer freely "
    "available and are approximated with fixed default values. Predictions "
    "should not be relied on for real security decisions."
)


@st.cache_resource
def load_model():
    model = joblib.load("phishing_model.pkl")
    columns = joblib.load("feature_columns.pkl")
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
    "Model trained on the UCI Phishing Websites dataset (11,055 labeled URLs, 30 features). "
    "See the accompanying notebook for training and evaluation details."
)

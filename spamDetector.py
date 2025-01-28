import streamlit as st
import pandas as pd
import pickle
import string

# Function to preprocess text
def preprocess_text(text):
    text = text.lower()  # Convert to lowercase
    text = ''.join([char for char in text if char not in string.punctuation])  # Remove punctuation
    return text

# Load the trained model and vectorizer
try:
    model = pickle.load(open("Spam.pkl", "rb"))
    vectorizer = pickle.load(open("vec.pkl", "rb"))
except Exception as e:
    st.error("Error loading model or vectorizer. Ensure 'Spam.pkl' and 'vec.pkl' are in the same directory.")
    st.stop()

# Streamlit App Layout
st.set_page_config(
    page_title="Spam SMS Classifier",
    page_icon="📧",
    layout="wide",
)

# Sidebar
with st.sidebar:
    st.title("📧 Spam SMS Classifier")
    st.markdown(
        """
        This app classifies sms text as:
        - **SPAM**: Unwanted or junk sms.
        - **HAM**: Legitimate or not spam.
        
        Powered by machine learning techniques.
        """
    )
    st.markdown("---")
    st.info("Developer: **VaishnaviVinod**")

# Main Section
st.title("Spam SMS Classifier")
st.markdown("### Enter the sms text below to classify it as Spam or Ham.")

# Input section
sms_input = st.text_area("SMS Text", height=200, placeholder="Type or paste the sms text here...")

# Predict Button
if st.button("🔍 Classify SMS"):
    if sms_input.strip() == "":
        st.warning("Please enter some text to classify.")
    else:
        # Preprocess the input
        processed_sms = preprocess_text(sms_input)
        
        # Transform the input using the vectorizer
        transformed_sms = vectorizer.transform([processed_sms])
        
        # Predict using the model
        prediction = model.predict(transformed_sms)[0]
        
        # Display the result
        if prediction == 1:
            st.error("🚨 This sms is classified as **SPAM**.")
        else:
            st.success("✅ This sms is classified as **HAM** (not spam).")

# Footer with subtle branding
st.markdown(
    """
    <style>
        footer {visibility: hidden;}
        .stApp {background-color: #f9f9f9;}
    </style>
    """,
    unsafe_allow_html=True,
)



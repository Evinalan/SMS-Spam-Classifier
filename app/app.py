
import streamlit as st
import sys
from pathlib import Path

# Add the main project directory to Python's import path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import prediction function
from src.predict import predict_sms


# Page configuration
st.set_page_config(
    page_title="SMS Spam Classifier",
    page_icon="📩",
    layout="centered"
)

# Application heading
st.title("📩 SMS Spam Classification System")

st.write(
    "Enter an SMS message below to check "
    "whether it is Spam or Ham."
)

# SMS input
message = st.text_area(
    "Enter your SMS message:",
    placeholder="Type your message here..."
)

# Prediction button
if st.button("Check Message"):

    if message.strip() == "":
        st.warning("Please enter an SMS message.")

    else:
        # Call the ML prediction function
        result = predict_sms(message)

        # Display prediction
        if str(result).lower() == "spam":
            st.error("🚨 This message is SPAM!")

        else:
            st.success("✅ This message is HAM!")


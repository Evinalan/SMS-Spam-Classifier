
import pandas as pd
import re
import nltk
import string

from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer


# STEP 1: Initialize NLP tools

stemmer = PorterStemmer()
import nltk

# Download required NLTK datasets
nltk.download("stopwords", quiet=True)
nltk.download("punkt", quiet=True)
nltk.download("punkt_tab", quiet=True)
stop_words = set(stopwords.words("english"))

# Preserve negation words
stop_words = stop_words - {"no", "not", "nor"}


# STEP 2: Define reusable preprocessing function

def preprocess_text(text):

    # Convert to lowercase
    text = text.lower()

    # Remove punctuation
    text = text.translate(
        str.maketrans("", "", string.punctuation)
    )

    # Remove numbers and special characters
    text = re.sub(r"[^a-zA-Z\s]", "", text)

    # Tokenization
    tokens = word_tokenize(text)

    # Remove stopwords
    tokens = [
        word for word in tokens
        if word not in stop_words
    ]

    # Perform stemming
    tokens = [
        stemmer.stem(word)
        for word in tokens
    ]

    # Convert tokens back into a sentence
    cleaned_text = " ".join(tokens)

    return cleaned_text


# STEP 3: Process the original dataset
# This section runs only when preprocess.py
# is executed directly, not when imported by Streamlit.

if __name__ == "__main__":

    df = pd.read_csv(
        "dataset/SMSSpamCollection",
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    df["original_message"] = df["message"].copy()

    print("Original Dataset:")
    print(df.head())

    # Apply preprocessing function
    df["cleaned_message"] = df["message"].apply(
        preprocess_text
    )

    print("\nOriginal vs Cleaned Messages:")

    print(
        df[
            ["label", "original_message", "cleaned_message"]
        ].head(10).to_string(index=False)
    )

    # Save cleaned dataset
    df[
        ["label", "original_message", "cleaned_message"]
    ].to_csv(
        "dataset/cleaned_sms.csv",
        index=False
    )

    print("\nCleaned dataset saved successfully!")

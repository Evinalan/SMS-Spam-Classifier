
import joblib

from src.preprocess import preprocess_text


# Load the trained ML model
model = joblib.load("models/best_model.pkl")

# Load the TF-IDF vectorizer
tfidf = joblib.load("models/tfidf_vectorizer.pkl")


def predict_sms(message):

    # Step 1: Preprocess the message
    cleaned_message = preprocess_text(message)

    # Step 2: Convert message into TF-IDF features
    message_vector = tfidf.transform([cleaned_message])

    # Step 3: Predict using trained model
    prediction = model.predict(message_vector)

    # Step 4: Return prediction
    return prediction[0]


from sklearn.feature_extraction.text import TfidfVectorizer


# Function to convert SMS messages into numerical features
def extract_features(X_train, X_test):

    # Create TF-IDF vectorizer
    tfidf = TfidfVectorizer(
        max_features=5000,
        min_df=2,
        ngram_range=(1, 2)
    )

    # Learn vocabulary and convert training messages
    X_train_tfidf = tfidf.fit_transform(X_train)

    # Convert testing messages using the same vocabulary
    X_test_tfidf = tfidf.transform(X_test)

    return X_train_tfidf, X_test_tfidf, tfidf

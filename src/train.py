
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from feature_extraction import extract_features


# STEP 1: Load cleaned dataset

data = pd.read_csv("dataset/cleaned_sms.csv")

print("Original dataset size:", len(data))


# STEP 2: Check missing values

print("\nMissing values before cleaning:")

print(data[["cleaned_message", "label"]].isnull().sum())


# STEP 3: Remove missing values

data = data.dropna(
    subset=["cleaned_message", "label"]
).copy()


# STEP 4: Convert values into strings

data["cleaned_message"] = (
    data["cleaned_message"].astype(str).str.strip()
)

data["label"] = (
    data["label"].astype(str).str.strip().str.lower()
)


# STEP 5: Remove empty messages and invalid labels

data = data[
    (data["cleaned_message"] != "") &
    (data["label"].isin(["spam", "ham"]))
].copy()


print("\nDataset size after cleaning:", len(data))


# STEP 6: Separate input and output

X = data["cleaned_message"]

y = data["label"]


# STEP 7: Split dataset

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y

)


print("\nTraining messages:", len(X_train))

print("Testing messages:", len(X_test))


# STEP 8: TF-IDF feature extraction

X_train_tfidf, X_test_tfidf, tfidf = extract_features(

    X_train,
    X_test

)


print("\nTF-IDF extraction completed!")


# STEP 9: Create machine learning models

models = {

    "Naive Bayes": MultinomialNB(),

    "Logistic Regression": LogisticRegression(
        max_iter=1000
    ),

    "SVM": LinearSVC()

}

# Create models directory if it doesn't exist

os.makedirs("models", exist_ok=True)
best_model = None

best_model_name = ""

best_f1_score = 0
# STEP 10: Train and evaluate models

for model_name, model in models.items():

    print("\n================================")

    print("MODEL:", model_name)

    # Train model
    model.fit(X_train_tfidf, y_train)

    # Predict testing data
    predictions = model.predict(X_test_tfidf)


    # Calculate metrics

    accuracy = accuracy_score(y_test, predictions)

    precision = precision_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    recall = recall_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        predictions,
        pos_label="spam",
        zero_division=0
    )

    # Select the model with the highest spam F1-score

    if f1 > best_f1_score:

      best_f1_score = f1

      best_model = model

      best_model_name = model_name
    # Display evaluation results

    print("\nAccuracy:", round(accuracy * 100, 2), "%")

    print("Precision:", round(precision * 100, 2), "%")

    print("Recall:", round(recall * 100, 2), "%")

    print("F1 Score:", round(f1 * 100, 2), "%")


    # Display confusion matrix

    print("\nConfusion Matrix:")

    cm = confusion_matrix(
        y_test,
        predictions,
        labels=["ham", "spam"]
    )

    print(cm)


    # Display classification report

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            labels=["ham", "spam"],
            target_names=["ham", "spam"],
            zero_division=0
        )
    )
# Save the best machine learning model

joblib.dump(
    best_model,
    "models/best_model.pkl"
)


# Save TF-IDF vectorizer

joblib.dump(
    tfidf,
    "models/tfidf_vectorizer.pkl"
)


print("\n================================")

print("BEST MODEL:", best_model_name)

print("BEST SPAM F1 SCORE:", round(best_f1_score * 100, 2), "%")

print("\nModel saved successfully!")

print("Saved files:")

print("models/best_model.pkl")

print("models/tfidf_vectorizer.pkl")

print("\nAll models evaluated successfully!")

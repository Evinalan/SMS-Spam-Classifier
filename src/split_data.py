
import pandas as pd

from sklearn.model_selection import train_test_split


# STEP 1: Load cleaned dataset

data = pd.read_csv("dataset/cleaned_sms.csv")


# STEP 2: Separate messages and labels

X = data["cleaned_message"]

y = data["label"]


# STEP 3: Split dataset into training and testing

X_train, X_test, y_train, y_test = train_test_split(

    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y

)


# STEP 4: Display dataset information

print("Dataset loaded successfully!")

print("\nTotal messages:", len(data))

print("Training messages:", len(X_train))

print("Testing messages:", len(X_test))


# STEP 5: Display training labels

print("\nTraining label distribution:")

print(y_train.value_counts())


# STEP 6: Display testing labels

print("\nTesting label distribution:")

print(y_test.value_counts())

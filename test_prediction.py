
from src.predict import predict_sms

messages = [
    "Congratulations! You won a free prize.",
    "Hey, how are you?",
    "Claim your free reward now!",
    "I will meet you tomorrow."
]

for message in messages:

    result = predict_sms(message)

    print("Message:", message)
    print("Prediction:", result)
    print("------------------------")


from src.predict import predict_sms


def test_spam_prediction():

    result = predict_sms(
        "Congratulations! You won a free prize."
    )

    assert str(result).lower() == "spam"


def test_ham_prediction():

    result = predict_sms(
        "Hey, how are you doing today?"
    )

    assert str(result).lower() == "ham"

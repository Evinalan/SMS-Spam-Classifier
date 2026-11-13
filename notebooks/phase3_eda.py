import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from wordcloud import WordCloud

from collections import Counter

import string
import re
# Load dataset

df = pd.read_csv(
    "dataset/SMSSpamCollection",
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display first five rows

print("First Five Records:")
print(df.head())
# Dataset shape

print("\nDataset Shape:")
print(df.shape)

# Dataset information

print("\nDataset Information:")
df.info()

# Statistical summary

print("\nDataset Description:")
print(df.describe())

# Check missing values

print("\nMissing Values:")
print(df.isnull().sum())
# Count spam and ham messages

plt.figure(figsize=(7, 5))

sns.countplot(
    data=df,
    x="label",
    hue="label",
    palette={"ham": "green", "spam": "red"},
    legend=False
)

plt.title("Spam vs Ham Message Distribution")
plt.xlabel("Message Type")
plt.ylabel("Number of Messages")

plt.show()
# Calculate message length

df["message_length"] = df["message"].apply(len)

# Display first five records

print("\nMessage Length:")
print(df[["label", "message", "message_length"]].head())

# Average message length

print("\nAverage Message Length:")
print(df.groupby("label")["message_length"].mean())
# Message length distribution

plt.figure(figsize=(10, 5))

sns.histplot(
    data=df,
    x="message_length",
    hue="label",
    bins=40,
    kde=True,
    element="step"
)

plt.title("SMS Message Length Distribution")
plt.xlabel("Message Length (Characters)")
plt.ylabel("Frequency")

plt.show()
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

# Convert text into lowercase and extract words

def get_words(text):

    text = text.lower()

    words = re.findall(r"\b[a-z]+\b", text)

    words = [
        word for word in words
        if word not in ENGLISH_STOP_WORDS
    ]

    return words


# Separate spam and ham messages

spam_messages = df[df["label"] == "spam"]["message"]

ham_messages = df[df["label"] == "ham"]["message"]


# Count spam words

spam_words = []

for message in spam_messages:
    spam_words.extend(get_words(message))

spam_word_count = Counter(spam_words)


# Count ham words

ham_words = []

for message in ham_messages:
    ham_words.extend(get_words(message))

ham_word_count = Counter(ham_words)


# Display most common words

print("\nMost Common Spam Words:")
print(spam_word_count.most_common(20))

print("\nMost Common Ham Words:")
print(ham_word_count.most_common(20))
# Get top 15 spam words

spam_top_words = spam_word_count.most_common(15)

words = [item[0] for item in spam_top_words]
counts = [item[1] for item in spam_top_words]

plt.figure(figsize=(12, 6))

sns.barplot(
    x=counts,
    y=words,
    hue=words,
    palette="Reds_r",
    legend=False
)

plt.title("Top 15 Most Common Words in Spam Messages")
plt.xlabel("Frequency")
plt.ylabel("Words")

plt.show()
# Get top 15 ham words

ham_top_words = ham_word_count.most_common(15)

words = [item[0] for item in ham_top_words]
counts = [item[1] for item in ham_top_words]

plt.figure(figsize=(12, 6))

sns.barplot(
    x=counts,
    y=words,
    hue=words,
    palette="Greens_r",
    legend=False
)

plt.title("Top 15 Most Common Words in Ham Messages")
plt.xlabel("Frequency")
plt.ylabel("Words")

plt.show()
# Combine all spam words

spam_text = " ".join(spam_words)

spam_wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    colormap="Reds"
).generate(spam_text)

plt.figure(figsize=(12, 6))

plt.imshow(spam_wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title("Spam Messages Word Cloud")

plt.show()
# Combine all ham words

ham_text = " ".join(ham_words)

ham_wordcloud = WordCloud(
    width=1000,
    height=500,
    background_color="white",
    colormap="Greens"
).generate(ham_text)

plt.figure(figsize=(12, 6))

plt.imshow(ham_wordcloud, interpolation="bilinear")

plt.axis("off")

plt.title("Ham Messages Word Cloud")

plt.show()
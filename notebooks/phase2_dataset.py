
import pandas as pd

# Dataset location
dataset_path = "dataset/SMSSpamCollection"

# Load dataset
df = pd.read_csv(
    dataset_path,
    sep="\t",
    header=None,
    names=["label", "message"]
)

# Display first 5 rows
print(df.head())
# Display dataset dimensions
print("\nDataset Shape:")
print(df.shape)

# Display total number of messages
print("\nTotal Messages:")
print(len(df))

# Count ham and spam messages
print("\nMessage Distribution:")
print(df["label"].value_counts())

# Display dataset information
print("\nDataset Information:")
print(df.info())

# Check missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Check duplicate messages
print("\nDuplicate Messages:")
print(df.duplicated().sum())
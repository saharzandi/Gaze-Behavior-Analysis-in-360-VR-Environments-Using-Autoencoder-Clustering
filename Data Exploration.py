import pandas as pd

# Define the file path for your dataset
file_path = "/Users/saharzandi/Documents/MDPI 2025 AR:VR/Dataset/Panonut360/Logs/user1/Alien.csv"

# Load the dataset into a Pandas DataFrame
try:
    dataset = pd.read_csv(file_path)
    print("Dataset loaded successfully!")
except Exception as e:
    print(f"Error loading dataset: {e}")

# Display basic information about the dataset
print("\nDataset Info:")
print(dataset.info())

# Display the first few rows of the dataset
print("\nFirst 5 Rows of the Dataset:")
print(dataset.head())

# Check for missing values in the dataset
print("\nMissing Values in Each Column:")
print(dataset.isnull().sum())

# Summary statistics for numerical columns
print("\nSummary Statistics:")
print(dataset.describe())

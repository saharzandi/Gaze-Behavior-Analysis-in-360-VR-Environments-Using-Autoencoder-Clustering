import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Define the file path for your dataset
file_path = "/Users/saharzandi/Documents/MDPI 2025 AR:VR/Dataset/Panonut360/Logs/user1/Alien.csv"

# Load the dataset
dataset = pd.read_csv(file_path)

# Convert 'player_time' to datetime format
dataset['player_time'] = pd.to_datetime(dataset['player_time'], errors='coerce')

# Normalize numerical columns
numerical_columns = ['head_x', 'head_y', 'head_z', 'eye_c_x', 'eye_c_y', 'eye_c_z',
                     'eye_x_t', 'eye_y_t', 'eye_z_t', 'head_pitch', 'head_yaw', 'head_roll',
                     'head_quaternion_w', 'head_quaternion_x', 'head_quaternion_y', 'head_quaternion_z',
                     'eye_quaternion_w', 'eye_quaternion_x', 'eye_quaternion_y', 'eye_quaternion_z']

scaler = MinMaxScaler()
dataset[numerical_columns] = scaler.fit_transform(dataset[numerical_columns])

# Save the processed dataset
processed_file_path = "/Users/saharzandi/Documents/MDPI 2025 AR:VR/Dataset/Panonut360/Logs/user1/Processed_Alien.csv"
dataset.to_csv(processed_file_path, index=False)

print("Preprocessing complete. Processed dataset saved at:")
print(processed_file_path)

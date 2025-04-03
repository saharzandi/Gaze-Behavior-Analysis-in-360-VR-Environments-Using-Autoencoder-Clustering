import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Path to dataset directory
input_dir = "/Users/saharzandi/Documents/MDPI 2025 AR:VR/Dataset/Panonut360/Logs"

# Process each user folder and their files
for user_folder in os.listdir(input_dir):
    user_path = os.path.join(input_dir, user_folder)
    if os.path.isdir(user_path):
        # Create a 'processed' directory inside the user's folder
        processed_dir = os.path.join(user_path, "processed")
        os.makedirs(processed_dir, exist_ok=True)

        for file in os.listdir(user_path):
            if file.endswith(".csv"):
                file_path = os.path.join(user_path, file)
                print(f"Processing {file_path}...")

                try:
                    # Load the dataset
                    dataset = pd.read_csv(file_path)

                    # Identify numeric columns
                    numeric_columns = dataset.select_dtypes(include=['float64', 'int64']).columns

                    # Handle non-numeric or missing values in numeric columns
                    dataset[numeric_columns] = dataset[numeric_columns].apply(pd.to_numeric, errors='coerce')
                    dataset[numeric_columns] = dataset[numeric_columns].fillna(dataset[numeric_columns].mean())

                    # Normalize numeric columns
                    scaler = MinMaxScaler()
                    dataset[numeric_columns] = scaler.fit_transform(dataset[numeric_columns])

                    # Save the processed dataset
                    output_file = f"Processed_{file}"
                    output_path = os.path.join(processed_dir, output_file)
                    dataset.to_csv(output_path, index=False)
                    print(f"Processed data saved at: {output_path}")

                    # Create and save the correlation matrix
                    correlation_matrix = dataset[numeric_columns].corr()
                    
                    # Save the correlation matrix as a CSV file
                    correlation_csv_path = os.path.join(processed_dir, f"Correlation_Matrix_{file.replace('.csv', '.csv')}")
                    correlation_matrix.to_csv(correlation_csv_path)
                    print(f"Correlation matrix CSV saved at: {correlation_csv_path}")

                    # Save the correlation matrix as a plot
                    corr_matrix_plot_path = os.path.join(processed_dir, f"Correlation_Matrix_{file.replace('.csv', '.png')}")
                    plt.figure(figsize=(10, 8))
                    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
                    plt.title(f"Correlation Matrix for {file}")
                    plt.savefig(corr_matrix_plot_path)
                    plt.close()
                    print(f"Correlation matrix plot saved at: {corr_matrix_plot_path}")

                    # Create and save the gaze positions plot
                    if set(["eye_x_t", "eye_y_t", "eye_z_t"]).issubset(dataset.columns):
                        gaze_plot_path = os.path.join(processed_dir, f"Gaze_Positions_{file.replace('.csv', '.png')}")
                        plt.figure(figsize=(12, 6))
                        plt.plot(dataset["eye_x_t"], label="eye_x_t", alpha=0.7)
                        plt.plot(dataset["eye_y_t"], label="eye_y_t", alpha=0.7)
                        plt.plot(dataset["eye_z_t"], label="eye_z_t", alpha=0.7)
                        plt.legend()
                        plt.xlabel("Time (Index)")
                        plt.ylabel("Gaze Positions (Normalized)")
                        plt.title("Gaze Positions Over Time")
                        plt.savefig(gaze_plot_path)
                        plt.close()
                        print(f"Gaze positions plot saved at: {gaze_plot_path}")

                except Exception as e:
                    # Log the error and continue processing other files
                    print(f"Error processing {file_path}: {e}")

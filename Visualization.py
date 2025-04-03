import pandas as pd
import matplotlib.pyplot as plt

# Reload the processed dataset
file_path = "/Users/saharzandi/Documents/MDPI 2025 AR:VR/Dataset/Panonut360/Logs/user1/processed/Processed_Alien.csv"
processed_dataset = pd.read_csv(file_path)

# Plot gaze positions (eye_x_t, eye_y_t, eye_z_t) over time
plt.figure(figsize=(12, 6))
plt.plot(processed_dataset.index, processed_dataset['eye_x_t'], label='eye_x_t', alpha=0.7)
plt.plot(processed_dataset.index, processed_dataset['eye_y_t'], label='eye_y_t', alpha=0.7)
plt.plot(processed_dataset.index, processed_dataset['eye_z_t'], label='eye_z_t', alpha=0.7)
plt.title("Gaze Positions Over Time")
plt.xlabel("Time (Index)")
plt.ylabel("Gaze Positions (Normalized)")
plt.legend()
plt.show()

# Analyze correlations between head and gaze movements
correlation_matrix = processed_dataset[['head_x', 'head_y', 'head_z',
                                        'eye_x_t', 'eye_y_t', 'eye_z_t']].corr()

# Display the correlation matrix
print("\nCorrelation Matrix:\n", correlation_matrix)
# Save the correlation matrix to a CSV file
correlation_matrix_path = "/Users/saharzandi/Documents/MDPI 2025 AR:VR/Dataset/Panonut360/Logs/user1/processed/Correlation_Matrix.csv"
correlation_matrix.to_csv(correlation_matrix_path)

print(f"Correlation matrix saved successfully at: {correlation_matrix_path}")

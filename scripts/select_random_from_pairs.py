import pandas as pd
import numpy as np
import os

# Define input and output file paths
input_path = "C_experiments/data/MM_balanced_dataset.csv"
output_path = "C_experiments/data/dataset_for_llms.csv"

# Check if the input file exists
if os.path.exists(input_path):
    # Load the CSV file
    df = pd.read_csv(input_path)

    # Ensure the number of rows is even
    if len(df) % 2 != 0:
        df = df.iloc[:-1]  # Drop the last row if the count is odd

    # Select one random row from each consecutive pair
    selected_rows = []
    for i in range(0, len(df), 2):
        selected_rows.append(df.iloc[i + np.random.randint(2)])

    # Create a new DataFrame from the selected rows
    new_df = pd.DataFrame(selected_rows)

    # Save the new DataFrame to a CSV file
    new_df.to_csv(output_path, index=False)
    success = True
else:
    success = False

success

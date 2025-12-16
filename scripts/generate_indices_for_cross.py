# run by:
# cd C:\Users\Jakub\Documents\GitHub\scope-ambliguity
# python C_experiments/data/generate_indices_for_cross.py

import pandas as pd
import numpy as np

# Parameters
num_indices = 440 # Total number of indices
num_folds = 5 # Number of folds for cross-validation
fold_size = num_indices // num_folds # Size of each fold

# Randomly shuffle indices and split into 5 folds
indices = np.random.permutation(num_indices)
folds = [indices[i * fold_size:(i + 1) * fold_size] for i in range(num_folds)]

# Create a DataFrame for better presentation and saving
df_folds = pd.DataFrame({f'Fold_{i+1}': fold for i, fold in enumerate(folds)})

# Save to CSV file
file_path = "C_experiments/data/folds_indices.csv"
df_folds.to_csv(file_path, index=False)

file_path
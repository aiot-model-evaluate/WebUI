import os
import pandas as pd
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def normalize_data(data, reverse=False, use_min_max=False):
    # Remove the column 'Jetson-Xanier-NX' if it exists
    if 'Jetson-Xanier-NX' in data.columns:
        data = data.drop(columns=['Jetson-Xanier-NX'])

    data = data.transpose()
    
    if use_min_max:
        # Min-Max normalization
        if reverse:
            normalized_data = 1 - (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        else:
            normalized_data = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
    else:
        # Map data to (-5, 5) then apply sigmoid normalization
        mapped_data = -3 + 6 * (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        if reverse:
            normalized_data = sigmoid( - mapped_data)
        else:
            normalized_data = sigmoid(mapped_data)
    
    normalized_data = normalized_data.transpose()
    print(use_min_max, normalized_data)
    return normalized_data

def process_folder(directory):
    # Identify the "smaller is better" metrics
    smaller_better = ['infer_average_inference_delay.csv', 'infer_power.csv', 'train_power.csv']
    # Identify the "larger is better" metrics
    larger_better = [
        'infer_energy_efficiency.csv', 'infer_throughput.csv',
        'train_energy_efficiency.csv', 'train_throughput.csv'
    ]

    norm_directory = os.path.join(directory, 'norm')
    if not os.path.exists(norm_directory):
        os.makedirs(norm_directory)

    for filename in os.listdir(directory):
        if filename.endswith('.csv') and (filename in smaller_better or filename in larger_better):
            file_path = os.path.join(directory, filename)
            data = pd.read_csv(file_path, index_col=0)
            reverse = filename in smaller_better
            use_min_max = False  # 'energy_efficiency' in filename
            normalized_data = normalize_data(data, reverse=reverse, use_min_max=use_min_max)
            normalized_data.to_csv(os.path.join(norm_directory, 'norm_' + filename))
            print(f'Normalized data for {filename} has been saved.')

directory_path = ".\..\Table"  # Adjust directory path as needed
process_folder(directory_path)

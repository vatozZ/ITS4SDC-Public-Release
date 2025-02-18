"""
This script is the first script that labeled-dataset files need to be feed into it.
It gets the dataset that has test files, and it merges into a single file.

Merged file stores 'road_points' and 'test_outcome'.

All test cases should be in .json file format.

16 November 2024.
"""

import argparse
from scipy.interpolate import interp1d
import json, os
import numpy as np
from numpy import arange
import pandas as pd
from tqdm import tqdm

parser = argparse.ArgumentParser(description='Test cases combiner. Provide dataset path.')
parser.add_argument('--path', type=str, help='Provide dataset folder name.')

args = parser.parse_args()

dataset_path = args.path

data_list = []

def adjust_array_size(array, target_size=197):
    
    if array.shape[0] == target_size:
        return array
    
    elif array.shape[0] > target_size:
        indices = np.linspace(0, array.shape[0] - 1, target_size, dtype=int)
        return array[indices]
    
    
    else:
        current_indices = np.linspace(0, array.shape[0] - 1, array.shape[0])
        target_indices = np.linspace(0, array.shape[0] - 1, target_size)

        interpolator_x = interp1d(current_indices, array[:, 0], kind='linear')
        interpolator_y = interp1d(current_indices, array[:, 1], kind='linear')

        interpolated_x = interpolator_x(target_indices)
        interpolated_y = interpolator_y(target_indices)

        return np.column_stack((interpolated_x, interpolated_y))

# iterate all files in the directory
for file in tqdm(os.listdir(dataset_path)):
    if file.endswith('.json'):
        data_dict = {}
        # open json file
        with open(dataset_path + '/' + file, 'r') as f:
            jsonfile = json.load(f)
            
            if 'interpolated_points' in jsonfile.keys():
                data_dict['road_points'] = adjust_array_size(np.array(jsonfile['interpolated_points']))
            
            elif 'interpolated_road_points' in jsonfile.keys():
                data_dict['road_points'] = adjust_array_size(np.array(jsonfile['interpolated_road_points']))
            
            data_dict['test_outcome'] = jsonfile['test_outcome']
        
        data_list.append(data_dict)

data_frame = pd.DataFrame(data_list)

# write out the dataframe as a JSON file.
data_frame.to_json('dataset_combined_' + str(data_frame.shape[0]) + '.json', orient='records', indent=2)


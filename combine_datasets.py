"""
This script is the first script that labeled-dataset files need to be feed into it.
It gets the dataset that has test files, and it merges into a single file.

Merged file stores 'road_points' and 'test_outcome'.

16 November 2024.
"""
from utils import adjust_array_size

files_path = r'C:\Users\monster\Desktop\dataset_10000'

invalid_file_path = r'C:\Users\monster\Desktop\dataset_10000'

import shutil
import json, os
import numpy as np
from shapely.geometry import LineString
from numpy import arange
import pandas as pd

invalid = 0

data_list = []
# iterate all files in the directory
for file in os.listdir(files_path):
    if file.endswith('.json'):
        data_dict = {}
        # open json file
        with open(files_path + '/' + file, 'r') as f:
            jsonfile = json.load(f)
            if jsonfile['is_valid'] == False:
                invalid += 1
                shutil.copy(files_path + '/' + file, invalid_file_path + '/')
                continue

            data_dict['road_points'] = adjust_array_size(np.array(jsonfile['interpolated_points']))
            data_dict['test_outcome'] = jsonfile['test_outcome']
        data_list.append(data_dict)

print("Number of invalid tests:", invalid)
print("Number of valid tests:", len(data_list))

data_frame = pd.DataFrame(data_list)

# write out the dataframe as a JSON file.
data_frame.to_json('dataset_' + str(data_frame.shape[0]) + '.json', orient='records', indent=2)

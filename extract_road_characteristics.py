"""
It reads a JSON file that contains road coordinates and corresponding labels (pass, fail).
It extracts the (segment_angles, segment_lengths) features for each road.
A segment angle is the difference between the angles of two consecutive segments.
A segment length is the Euclidean distance between two road coordinates.
It saves the features in a JSON file.

18 November 2024.
@vatozZ
"""

import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='Feature Extractor.')
parser.add_argument('--test_suite_name', type=str, help='Provide test suite file name.')

args = parser.parse_args()

test_suite_filename = args.test_suite_name

with open(test_suite_filename, 'r') as f:
    jsondata = json.load(f)

segment_angles_all_list = []
segment_lengths_all_list = []
label_list = []

for test_case in jsondata:

    roadPointsArray = np.array(test_case['road_points'])

    # 1) Calculate segment angles
    dx = roadPointsArray[1:, 0] - roadPointsArray[:-1, 0]
    dy = roadPointsArray[1:, 1] - roadPointsArray[:-1, 1]

    raw_angles = np.rad2deg(np.arctan2(dy, dx))
    segment_angles = np.zeros_like(raw_angles)
    segment_angles[1:] = np.diff(raw_angles)

    segment_angles_all_list.append(list(segment_angles))

    # 2) Calculate segment lengths

    _differences = roadPointsArray[1:] - roadPointsArray[:-1]
    segment_lengths = list(np.linalg.norm(_differences, axis=1))

    segment_lengths_all_list.append(segment_lengths)

    # 3) Assign labels
    
    if test_case['test_outcome'] == 'FAIL': label_list.append(0)
    elif test_case['test_outcome'] == 'PASS': label_list.append(1)
    else: quit("ERROR!")

with open(test_suite_filename.split('.json')[0] + '_road_characteristics.json', 'w') as f:
    json.dump(
        {'segment_angles': segment_angles_all_list,
         'segment_lengths': segment_lengths_all_list,
         'labels': label_list}, f, indent=4)

print("Road features are extracted successfully!")

"""
Model training for the competition. 
The model utilizes the full dataset. 
The dataset contains 10,000 test cases.
NB! There is no validation, as the model needs as much data as possible.
"""


#import pandas as pd
#from sklearn.model_selection import KFold
import os, json
import matplotlib.pyplot as plt
import numpy as np
#from sklearn.model_selection import train_test_split
#from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
#from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
#from tensorflow.keras.callbacks import LambdaCallback
#from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time
#from tensorflow.keras.callbacks import EarlyStopping


print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

with open('executed-dataset-10000_road_characteristics.json', 'r') as f:
    data = json.load(f) #data is a list of dict data typed variable.

angle_data = data['segment_angles']
length_data = data['segment_lengths']
label_data = data['labels']

X_data = []
for angles, lengths in zip(angle_data, length_data):
    segment_features = np.column_stack((angles, lengths))
    X_data.append(segment_features)

X = np.array(X_data)
y = np.array(label_data)

t0 = time.time()

print("Training has started!")

X_train = X
y_train = y

# Model initialization
model = Sequential()
model.add(Bidirectional(LSTM(units=220, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)))
model.add(Dense(units=1, activation='sigmoid'))

# Compile the model
optimizer = Adam(learning_rate=1e-3)
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, batch_size=1024, epochs=400, verbose=1, callbacks=[])

model.save(f"model_full_10000.h5")

print("Training has finished. Total training took", time.time() - t0, " second")
# Log all fold results to CSV

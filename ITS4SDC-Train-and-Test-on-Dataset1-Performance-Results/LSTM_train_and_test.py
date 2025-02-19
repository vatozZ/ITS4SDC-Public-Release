"""
Model training using k=10 fold cross validation.
Performance metrics will be stored in the cross_validation_results.csv file.

Table on the paper: Table5: "ITS4SDC’s Performance Metrics Across Setups".

Train and validate on Dataset1 (10000 test cases).
"""


import pandas as pd
from sklearn.model_selection import KFold
import os, json
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import precision_score, recall_score, f1_score
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Bidirectional
from tensorflow.keras.callbacks import LambdaCallback
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
import tensorflow as tf
import time
from tensorflow.keras.callbacks import EarlyStopping


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

# K-Fold Cross Validation
k = 10
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_num = 1
performance_metrics = [] # performance metrics will be stored.

# Confusion matrices will be stored under the created directory.
os.makedirs("confusion_matrices", exist_ok=True)

t0 = time.time()

print("Training has started!")
# Cross-validation loop
for train_index, test_index in kf.split(X):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Model initialization
    model = Sequential()
    model.add(Bidirectional(LSTM(units=220, input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=False)))
    model.add(Dense(units=1, activation='sigmoid'))

    # Compile the model
    optimizer = Adam(learning_rate=1e-3)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Custom callback to stop training if validation accuracy reaches 87%
    class CustomEarlyStopping(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            val_accuracy = logs.get("val_accuracy")
            if val_accuracy is not None and val_accuracy >= 0.88:
                print(f"\nFold {fold_num} reached 88% validation accuracy. Stopping early.")
                self.model.stop_training = True

    # Train the model
    model.fit(X_train, y_train, batch_size=1024, epochs=400, validation_data=(X_test, y_test), verbose=1, callbacks=[CustomEarlyStopping()])

    # Predictions and confusion matrix
    y_pred = (model.predict(X_test) > 0.5).astype("int32")
    cm = confusion_matrix(y_test, y_pred)

    # Save confusion matrix as image
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.figure()
    disp.plot()
    plt.title(f"Confusion Matrix - Fold {fold_num}")
    cm_filename = f'confusion_matrices/confusion_matrix_fold_{fold_num}.png'
    plt.savefig(cm_filename, dpi=1100, bbox_inches='tight')
    plt.close()

    # Calculate performance metrics
    accuracy = (cm[0, 0] + cm[1, 1]) / np.sum(cm) 
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Store metrics
    performance_metrics.append({
        'fold': fold_num,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix_file': cm_filename
    })

    print(
        f"Fold {fold_num} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}")

    fold_num += 1

    model.save(f"model_fold_{fold_num}.h5")

print("Training has finished. Total training took", time.time() - t0, " second")
# Log all fold results to CSV
metrics_df = pd.DataFrame(performance_metrics)
metrics_df.to_csv("cross_validation_results.csv", index=False)
print("Cross-validation results saved to 'cross_validation_results.csv'")

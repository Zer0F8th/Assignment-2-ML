import os
import pickle
import numpy as np
import pandas as pd

import talos as ta

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical

from sklearn.model_selection import train_test_split



###############################################################################
# 1. LOAD AND PREPARE CIFAR-100 DATA
###############################################################################

# Load training data
# CIFAR-100 dataset is stored in binary files, we need to load them using pickle
cifar_100_dir = 'cifar-100-python/'  # Change this to your CIFAR-100 directory
with open(cifar_100_dir + 'train', 'rb') as f:
    train_dict = pickle.load(f, encoding='latin1')
X_train = train_dict['data']  # shape: (50,000, 3072)
y_train = np.array(train_dict['fine_labels'])  # shape: (50,000,)

# Load test data
with open(cifar_100_dir + 'test', 'rb') as f:
    test_dict = pickle.load(f, encoding='latin1')
X_test = test_dict['data']  # shape: (10,000, 3072)
y_test = np.array(test_dict['fine_labels'])  # shape: (10,000,)

# Convert to float32 and scale pixels to [0,1]
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# Convert labels to one-hot vectors for 100 classes
y_train = to_categorical(y_train, 100)
y_test  = to_categorical(y_test, 100)

# Create validation split from training data
# Split the training data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

print("Data shapes:")
print("  X_train:", X_train.shape, "y_train:", y_train.shape)
print("  X_val:  ", X_val.shape,   "y_val:  ", y_val.shape)
print("  X_test: ", X_test.shape,  "y_test: ", y_test.shape)

###############################################################################
# 2. DEFINE HYPERPARAMETER SPACE FOR TALOS
###############################################################################

p = {
    'units': [120, 240],
    'hidden_activations': ['relu', 'sigmoid'],
    'activation': ['softmax', 'sigmoid'],      # output layer activation
    'loss': ['mse', 'categorical_crossentropy'],
    'optimizer': ['adam', 'adagrad'],
    'batch_size': [1000, 2000]
}

###############################################################################
# 3. BUILD THE MODEL FUNCTION FOR TALOS
###############################################################################

def cifar100_model(x_train, y_train, x_val, y_val, params):
    """
    Builds and trains a Keras Sequential model for CIFAR-100 classification
    according to the given 'params' dictionary from Talos.
    """
    model = Sequential()
    
    # Input layer
    model.add(Dense(params['units'], 
                    activation=params['hidden_activations'], 
                    input_shape=(3072,)))
    
    # 4. At least 5 hidden layers (here we do exactly 5)
    for _ in range(4):
        model.add(Dense(params['units'], 
                        activation=params['hidden_activations']))
    
    # Output layer with 100 classes
    model.add(Dense(100, activation=params['activation']))
    
    # Compile the model
    model.compile(optimizer=params['optimizer'],
                  loss=params['loss'],
                  metrics=['accuracy'])
    
    # 7. Train for 200 epochs
    out = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=200,
        batch_size=params['batch_size'],
        verbose=0  # change to 1 or 2 for more detailed logging
    )
    
    # Return training history and the model
    return out, model

###############################################################################
# 4. RUN THE TALOS HYPERPARAMETER SCAN
###############################################################################
# This will run 64 permutations, each for 200 epochs -> very time-consuming!

scan_object = ta.Scan(
    x=X_train,
    y=y_train,
    x_val=X_val,
    y_val=y_val,
    params=p,
    model=cifar100_model,
    experiment_name='cifar100_experiment',
    print_params=True  # prints the parameters for each run
)

###############################################################################
# 5. SAVE TALOS RESULTS TO CSV
###############################################################################

# Create output folder if it doesn't exist
os.makedirs('talos_output', exist_ok=True)

# Retrieve the DataFrame of results
results_df = scan_object.data

# Save to CSV
results_csv = os.path.join('talos_output', 'cifar100_results.csv')
results_df.to_csv(results_csv, index=False)

print(f"\nTalos scan complete! Results saved to:\n  {os.path.abspath(results_csv)}")




# README

This repository contains a Python script demonstrating how to build and tune a multilayer perceptron (MLP) for classifying images from the **CIFAR-100** dataset using **Keras** and **Talos**. The script:

1. **Loads** the CIFAR-100 dataset (from binary files)  
2. **Preprocesses** the images (normalizes pixel values) and converts labels to one-hot form  
3. **Splits** the dataset into training, validation, and test sets  
4. **Defines** a parameter dictionary for Talos hyperparameter tuning  
5. **Builds** a Keras `Sequential` model with at least five hidden layers  
6. **Trains** the model on each hyperparameter combination for 200 epochs  
7. **Saves** the Talos tuning results to a CSV file  

## Contents

- **`main.py`** (or your chosen filename): Contains the complete code to load CIFAR-100 data, perform model training, run hyperparameter scans with Talos, and save results.

## Requirements


1. **Python 3.7+** (Talos often works best in Python 3.7–3.9 environments)
2. **Dependencies** (typical packages):
   - `numpy`
   - `pandas`
   - `scikit-learn`
   - `tensorflow`
   - `talos`
3. cifar-100-python dataset

You must download the [cifar-100-python](https://www.cs.toronto.edu/~kriz/cifar.html) and place it at top level of the project under the `cifar-100-python/` directory.

## Setup

To get the project up and running, it’s recommended to work inside a Python virtual environment. This isolates your project’s dependencies and keeps your system clean.

First, create the virtual environment:

```bash
python3 -m venv .
```

Then activate it (your shell prompt should change to indicate the environment is active):

```bash
source bin/activate
```

Next, set the required environment variable to allow installation of deprecated scikit‑learn packages:

```bash
export SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
```

Finally, install all necessary dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

## Files and Folders

- **`cifar-100-python/`**  
  - Contains the **binary files**: `train` and `test`. These files hold the CIFAR-100 images and labels as pickled dictionaries.
- **`talos_output/`**  
  - Created automatically. Stores the results of the hyperparameter tuning as a CSV file (named `cifar100_results.csv` in the example).

## How to Run

1. **Ensure** the `cifar-100-python` directory (with `train` and `test` files) is available in the same folder as the script or update the `cifar_100_dir` variable in the code to point to the correct path.
2. **Install** dependencies:
   ```bash
   pip install -r requirments.txt
   ```
3. **Execute** the script:
   ```bash
   python main.py
   ```
   - The code will:
     1. Load CIFAR-100 data
     2. Split into training, validation, and test sets
     3. Define a parameter dictionary for Talos
     4. Build a Keras model with 5 hidden Dense layers
     5. Perform a Talos hyperparameter scan (64 combinations)
     6. Train each model for 200 epochs  
     7. Save results into `talos_output/cifar100_results.csv`
4. **Wait** for the scan to complete. This step can be **time-consuming** because each of the 64 permutations runs for 200 epochs.

## Code Breakdown

1. **Import Modules**  
   The script imports essential libraries:
   - `pickle` for loading binary data  
   - `numpy` for numerical operations  
   - `pandas` for saving the results in a CSV  
   - `talos` for hyperparameter tuning  
   - `tensorflow.keras` for building the neural network model  
   - `train_test_split` from `sklearn.model_selection` for creating a validation set

2. **Load CIFAR-100 Data**  
   - Reads the `train` and `test` binary files using `pickle`  
   - Extracts image data (`X_train`, `X_test`) and labels (`y_train`, `y_test`)  
   - Reshapes the data as needed (in this code, the data is already flattened to 3072 features: 32×32×3)  
   - Normalizes pixel values to the range [0,1]  
   - Converts labels to one-hot format with 100 classes  

3. **Create a Validation Set**  
   - Uses `train_test_split` to reserve 20% of the training data as a **validation set** (`X_val`, `y_val`)

4. **Define Hyperparameter Space (`p`)**  
   - Specifies possible values for:
     - `units` (number of neurons in each layer)  
     - `hidden_activations` (activation for hidden layers)  
     - `activation` (activation for the output layer)  
     - `loss` (loss function)  
     - `optimizer` (training optimizer)  
     - `batch_size` (batch size for training)

5. **Build the Keras Model (`cifar100_model`)**  
   - Creates a Keras `Sequential` model  
   - **Input layer**: `params['units']` neurons, activation `params['hidden_activations']`  
   - **Hidden layers**: 4 more dense layers, each with `params['units']` and `params['hidden_activations']` (for a total of at least 5 hidden layers)  
   - **Output layer**: 100 neurons (for 100 classes) with `params['activation']`  
   - Compiles the model using `params['optimizer']` and `params['loss']`  
   - Trains for **200 epochs**, with optional verbose output

6. **Run Talos Scan**  
   - **`ta.Scan()`** loops over **all 64** hyperparameter permutations  
   - Each permutation trains the model for 200 epochs  
   - Collects metrics (loss, accuracy, etc.) for each run

7. **Save Results**  
   - Creates a folder **`talos_output/`** if it doesn’t exist  
   - Saves a CSV (`cifar100_results.csv`) containing each hyperparameter set and corresponding results

## Notes

- **Runtime**: Because it runs 64 models × 200 epochs each, it can take a **long** time on CPU. Consider running on a GPU or reducing the number of permutations/epochs if you want faster experimentation.
- **Accuracy**: An MLP may not reach extremely high accuracy on CIFAR-100 (it’s better tackled by convolutional neural networks). Expect modest results in comparison to CNNs.
- **Customization**: You can adjust layer count, epochs, or hyperparameter values in the script to experiment with different architectures or training setups.

## Contributing / Modifications

- **Change the dataset path**: Update the variable `cifar_100_dir` if your CIFAR-100 files are in a different location.
- **Adjust hyperparameters**: Add or remove items from `p` to reduce or expand the grid search.
- **Modify layers**: You can increase or decrease the number of hidden layers in the `for _ in range(4)` loop.
- **Verbose**: You can set `verbose=1` or `verbose=2` in `model.fit()` for more detailed training logs.

---

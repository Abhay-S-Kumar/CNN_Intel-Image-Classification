# CNN_Intel-Image-Classification
# Intel Image Classification using Convolutional Neural Networks (CNN)

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify images from the Intel Image Classification dataset. The goal is to categorize images into six distinct classes: `buildings`, `forest`, `glacier`, `mountain`, `sea`, and `street`.

## Dataset
The dataset used is the "Intel Image Classification" dataset, downloaded from Kagglehub. It consists of a large collection of images categorized into the six classes mentioned above.

**Dataset Source**: [Kagglehub - Intel Image Classification](https://www.kaggle.com/datasets/puneet6060/intel-image-classification)

The dataset is split into training, validation, and test sets. Data augmentation is applied to the training set to improve model generalization.

## Model Architecture
The model is a sequential CNN built with TensorFlow/Keras. It includes:
-   A `Rescaling` layer to normalize pixel values from `[0, 255]` to `[0, 1]`.
-   Multiple `Conv2D` layers with `relu` activation for feature extraction.
-   `BatchNormalization` layers after each `Conv2D` layer to stabilize and speed up training.
-   `MaxPooling2D` layers to reduce dimensionality.
-   A `Flatten` layer to convert the 2D feature maps into a 1D vector.
-   `Dropout` layers (with a rate of 0.5) before the dense layers to prevent overfitting.
-   `Dense` layers with `relu` activation.
-   An output `Dense` layer with `softmax` activation for multi-class classification.

## Training
-   **Optimizer**: Adam
-   **Loss Function**: `sparse_categorical_crossentropy`
-   **Metrics**: `accuracy`
-   **Epochs**: 50 (with Early Stopping)
-   **Callbacks**: `EarlyStopping` is used to monitor `val_loss` with a `patience` of 5 epochs, restoring the best weights, and a `min_delta` of 0.001 to prevent marginal improvements from stopping training too early.

## Results
After training, the model's performance is evaluated on a dedicated test set. Key metrics and visualizations include:
-   **Test Accuracy**: (The latest test accuracy observed was around 70.12%, but this will vary with re-runs and model improvements).
-   **Training and Validation Accuracy/Loss Plots**: Visual representation of model performance over epochs.
-   **Confusion Matrix**: A heatmap showing the counts of correct and incorrect predictions for each class on the validation set.
-   **Sample Predictions**: Visualization of actual vs. predicted classes with confidence scores for a few images from the validation set.

## Setup and Usage

### Requirements
To run this notebook, you will need the following libraries:
-   `numpy`
-   `pandas`
-   `matplotlib`
-   `tensorflow`
-   `kagglehub`
-   `scikit-learn`
-   `seaborn`

These can typically be installed via `pip`:
```bash
pip install numpy pandas matplotlib tensorflow kagglehub scikit-learn seaborn
```

### Running the Code
1.  **Download the Notebook**: Clone this repository or download the `.ipynb` file.
2.  **Open in Colab**: Open the notebook in Google Colaboratory.
3.  **Run Cells**: Execute the cells sequentially. The notebook will automatically download the dataset from Kagglehub.
    -   The `kagglehub.dataset_download` function handles dataset download and local path creation.
    -   The model will be trained, evaluated, and various plots will be generated.

## Code Structure
-   **Imports**: Essential libraries are imported at the beginning.
-   **Data Loading**: The dataset is downloaded and loaded using `tf.keras.preprocessing.image_dataset_from_directory`.
-   **Data Exploration**: Sample images are displayed.
-   **Data Preprocessing**: Data augmentation is applied.
-   **Model Definition**: The CNN architecture is defined.
-   **Model Compilation**: The model is compiled with an optimizer, loss function, and metrics.
-   **Model Training**: The model is trained using the `fit` method with Early Stopping.
-   **Model Evaluation**: The model is evaluated on the test set.
-   **Visualization**: Plots for accuracy/loss, confusion matrix, and sample predictions are generated.

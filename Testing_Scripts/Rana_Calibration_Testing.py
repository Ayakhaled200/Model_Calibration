import os
import sys
import django
import httpx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import balanced_accuracy_score

# Set up the Django environment
django_project_path = r'E:\Giza Systems\Calibration\Rana Hossny\rana\prediction_project'
sys.path.append(django_project_path)
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'prediction_project.settings')

try:
    django.setup()
except Exception as e:
    print(f"Django setup failed: {e}")

# Paths for training and test data
train_data_path = "E:\Giza Systems\Calibration\Knowledge_base_train.csv"
test_data_path = "E:\Giza Systems\Calibration\Knowledge_base_test.csv"

# Load training and test data
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# # Function to group unique values in the target class
# def group_unique_values(dataset, label_column):
#     unique_values = dataset[label_column].value_counts()
#     return unique_values
#
# train_unique_values = group_unique_values(train_df, "class")
# test_unique_values = group_unique_values(test_df, "class")
#
# print("data distribution in train dataset", train_unique_values)
# print("data distribution in test dataset", test_unique_values)

# Extract features and true labels for train and test
train_features = train_df.drop(columns=["class"])
train_labels = train_df["class"]

test_features = test_df.drop(columns=["class"])  
test_labels = test_df["class"]

# Function to transform input field names
def transform_field_names(data):
    """Convert field names to match the required server format."""
    # Define a mapping from original field names to required server field names
    field_name_mapping = {
        "Sum of Instances in Clients": "sum_of_instances_in_clients",
        "Max. Of Instances in Clients": "max_of_instances_in_clients",
        "Min. Of Instances in Clients": "min_of_instances_in_clients",
        "Stddev of Instances in Clients": "stddev_of_instances_in_clients",
        "Average Dataset Missing Values %": "average_dataset_missing_values_percent",
        "Min Dataset Missing Values %": "min_dataset_missing_values_percent",
        "Max Dataset Missing Values %": "max_dataset_missing_values_percent",
        "Stddev Dataset Missing Values %": "stddev_dataset_missing_values_percent",
        "Average Target Missing Values %": "average_target_missing_values_percent",
        "Min Target Missing Values %": "min_target_missing_values_percent",
        "Max Target Missing Values %": "max_target_missing_values_percent",
        "Stddev Target Missing Values %": "stddev_target_missing_values_percent",
        "No. Of Features": "no_of_features",
        "No. Of Numerical Features": "no_of_numerical_features",
        "No. Of Categorical Features": "no_of_categorical_features",
        "Sampling Rate": "sampling_rate",
        "Average Skewness of Numerical Features": "average_skewness_of_numerical_features",
        "Minimum Skewness of Numerical Features": "minimum_skewness_of_numerical_features",
        "Maximum Skewness of Numerical Features": "maximum_skewness_of_numerical_features",
        "Stddev Skewness of Numerical Features": "stddev_skewness_of_numerical_features",
        "Average Kurtosis of Numerical Features": "average_kurtosis_of_numerical_features",
        "Minimum Kurtosis of Numerical Features": "minimum_kurtosis_of_numerical_features",
        "Maximum Kurtosis of Numerical Features": "maximum_kurtosis_of_numerical_features",
        "Stddev Kurtosis of Numerical Features": "stddev_kurtosis_of_numerical_features",
        "Avg No. of Symbols per Categorical Features": "avg_no_of_symbols_per_categorical_features",
        "Min. No. Of Symbols per Categorical Features": "min_no_of_symbols_per_categorical_features",
        "Max. No. Of Symbols per Categorical Features": "max_no_of_symbols_per_categorical_features",
        "Stddev No. Of Symbols per Categorical Features": "stddev_no_of_symbols_per_categorical_features",
        "Avg No. Of Stationary Features": "avg_no_of_stationary_features",
        "Min No. Of Stationary Features": "min_no_of_stationary_features",
        "Max No. Of Stationary Features": "max_no_of_stationary_features",
        "Stddev No. Of Stationary Features": "stddev_no_of_stationary_features",
        "Avg No. Of Stationary Features after 1st order": "avg_no_of_stationary_features_after_1st_order",
        "Min No. Of Stationary Features after 1st order": "min_no_of_stationary_features_after_1st_order",
        "Max No. Of Stationary Features after 1st order": "max_no_of_stationary_features_after_1st_order",
        "Stddev No. Of Stationary Features after 1st order": "stddev_no_of_stationary_features_after_1st_order",
        "Avg No. Of Stationary Features after 2nd order": "avg_no_of_stationary_features_after_2nd_order",
        "Min No. Of Stationary Features after 2nd order": "min_no_of_stationary_features_after_2nd_order",
        "Max No. Of Stationary Features after 2nd order": "max_no_of_stationary_features_after_2nd_order",
        "Stddev No. Of Stationary Features after 2nd order": "stddev_no_of_stationary_features_after_2nd_order",
        "Avg No. Of Significant Lags in Target": "avg_no_of_significant_lags_in_target",
        "Min No. Of Significant Lags in Target": "min_no_of_significant_lags_in_target",
        "Max No. Of Significant Lags in Target": "max_no_of_significant_lags_in_target",
        "Stddev No. Of Significant Lags in Target": "stddev_no_of_significant_lags_in_target",
        "Avg No. Of Insignificant Lags in Target": "avg_no_of_insignificant_lags_in_target",
        "Max No. Of Insignificant Lags in Target": "max_no_of_insignificant_lags_in_target",
        "Min No. Of Insignificant Lags in Target": "min_no_of_insignificant_lags_in_target",
        "Stddev No. Of Insignificant Lags in Target": "stddev_no_of_insignificant_lags_in_target",
        "Avg. No. Of Seasonality Components in Target": "avg_no_of_seasonality_components_in_target",
        "Max No. Of Seasonality Components in Target": "max_no_of_seasonality_components_in_target",
        "Min No. Of Seasonality Components in Target": "min_no_of_seasonality_components_in_target",
        "Stddev No. Of Seasonality Components in Target": "stddev_no_of_seasonality_components_in_target",
        "Average Fractal Dimensionality Across Clients of Target": "average_fractal_dimensionality_across_clients_of_target",
        "Maximum Period of Seasonality Components in Target Across Clients": "maximum_period_of_seasonality_components_in_target_across_clients",
        "Minimum Period of Seasonality Components in Target Across Clients": "minimum_period_of_seasonality_components_in_target_across_clients",
        "Entropy of Target Stationarity": "entropy_of_target_stationarity"
    }

    # Apply the mapping to each instance
    transformed_data = {}
    for instance_id, features in data.items():
        transformed_features = {
            field_name_mapping.get(k, k): v for k, v in features.items()
        }
        transformed_data[instance_id] = transformed_features

    return transformed_data


# Transform train and test data
train_data = transform_field_names({
    str(idx): row.to_dict() for idx, row in train_features.iterrows()
})
test_data = transform_field_names({
    str(idx): row.to_dict() for idx, row in test_features.iterrows()
})


# Send data to the server using httpx
url = "http://127.0.0.1:8000/predict/predict/"
async def send_request(data):
    async with httpx.AsyncClient(timeout=10) as client:
        response = await client.post(url, json=data)
        return response.json()

# Retrieve predictions for train and test
import asyncio
train_predictions = asyncio.run(send_request(train_data))
test_predictions = asyncio.run(send_request(test_data))

# Function to parse predictions
def parse_predictions(predictions, true_labels):
    probabilities = []  # Predicted probabilities for all classes
    classes = list(predictions["0"].keys())  # Extract class names from the first prediction
    y_true = []  # True labels

    for instance_id, prediction in predictions.items():
        y_true.append(true_labels[int(instance_id)])  # Append the true label
        probabilities.append([prediction[cls] for cls in classes])  # Class probabilities

    return y_true, probabilities, classes

# Parse training and test predictions
y_true_train, train_probabilities, classes = parse_predictions(train_predictions, train_labels)
y_true_test, test_probabilities, _ = parse_predictions(test_predictions, test_labels)

# Function to calculate total ECE
def calculate_total_ece(y_true, probabilities, classes, n_bins=10):
    """
    Calculate the total Expected Calibration Error (ECE) for the dataset.

    Args:
        y_true (list): True class labels.
        probabilities (list): Predicted probabilities for each class.
        classes (list): List of class labels.
        n_bins (int): Number of bins for calibration curve.

    Returns:
        float: Total ECE score for the dataset.
    """
    total_ece = 0
    total_samples = len(y_true)

    for idx, cls in enumerate(classes):
        # Create binary labels for one-vs-rest
        y_true_binary = np.array([1 if label == cls else 0 for label in y_true])
        class_probabilities = np.array([prob[idx] for prob in probabilities])

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_true_binary, class_probabilities, n_bins=n_bins, strategy="uniform")

        # Compute ECE for this class
        class_ece = np.sum(np.abs(prob_true - prob_pred) * len(prob_pred))
        total_ece += class_ece

    # Normalize by total samples
    total_ece /= total_samples
    return total_ece

# Updated evaluate_and_plot function
def evaluate_and_plot(y_true, probabilities, classes, dataset_name):
    # Convert probabilities to predictions
    y_pred = [classes[np.argmax(prob)] for prob in probabilities]

    # Recall, F1
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted")

    # Balanced Accuracy
    balanced_acc = balanced_accuracy_score(y_true, y_pred)

    print(f"\nMetrics for {dataset_name}:")
    print(f"Recall (Weighted): {recall:.4f}")
    print(f"F1 Score (Weighted): {f1:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")

    # Reliability Diagram
    plt.figure(figsize=(10, 8))

    for idx, cls in enumerate(classes):
        # Create binary labels for one-vs-rest
        y_true_binary = [1 if label == cls else 0 for label in y_true]
        class_probabilities = [prob[idx] for prob in probabilities]  # Probabilities for the current class

        # Compute calibration curve
        prob_true, prob_pred = calibration_curve(y_true_binary, class_probabilities, n_bins=10, strategy="uniform")

        # Plot calibration curve for this class
        plt.plot(prob_pred, prob_true, marker="o", label=f"Class {cls}")

    # Plot diagonal for perfect calibration
    plt.plot([0, 1], [0, 1], linestyle="--", label="Perfect Calibration", color="gray")

    # Add plot labels and legend
    plt.xlabel("Predicted Probability")
    plt.ylabel("True Frequency")
    plt.title(f"Reliability Diagram ({dataset_name})")
    plt.legend()
    plt.show()

    # Calculate and print total ECE
    total_ece = calculate_total_ece(y_true, probabilities, classes)
    print(f"\nTotal Expected Calibration Error (ECE) for {dataset_name}: {total_ece:.4f}")


# Evaluate and plot for training and test data
evaluate_and_plot(y_true_train, train_probabilities, classes, "Training Data")
evaluate_and_plot(y_true_test, test_probabilities, classes, "Test Data")

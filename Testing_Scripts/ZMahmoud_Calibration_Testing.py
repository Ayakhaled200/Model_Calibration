import httpx
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import recall_score, f1_score
from sklearn.calibration import calibration_curve
from sklearn.metrics import balanced_accuracy_score
import asyncio


# Paths for training and test data
train_data_path = "E:\Giza Systems\Calibration\Knowledge_base_train.csv"  # Path to training data CSV
test_data_path = "E:\Giza Systems\Calibration\Knowledge_base_test.csv"   # Path to test data CSV

# Load training and test data
train_df = pd.read_csv(train_data_path)
test_df = pd.read_csv(test_data_path)

# Extract features and true labels for train and test
train_features = train_df.drop(columns=["class"])
train_labels = train_df["class"]

test_features = test_df.drop(columns=["class"])
test_labels = test_df["class"]

# Prepare data for server (convert to dictionary with instance IDs as keys)
train_data = {
    str(idx): row.to_dict() for idx, row in train_features.iterrows()
}
test_data = {
    str(idx): row.to_dict() for idx, row in test_features.iterrows()
}

# Send data to the server using httpx
url = "http://127.0.0.1:5000/predict"  # Adjust based on your server's address
async def send_request(data):
    async with httpx.AsyncClient() as client:
        response = await client.post(url, json=data)
        return response.json()

# Retrieve predictions for train and test
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

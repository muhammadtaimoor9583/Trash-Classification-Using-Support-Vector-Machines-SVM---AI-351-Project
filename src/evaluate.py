import os
import joblib
# from train_svm import train_svm
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Evaluation function
def evaluate_model(model_path, X_test, y_test, classes):
    # Load the model
    svm, _ = joblib.load(model_path)

    # Predict
    y_pred = svm.predict(X_test)

    # Classification Report
    report = classification_report(y_test, y_pred, target_names=classes)
    print("Classification Report:\n", report)
    with open("results/classification_report.txt", "w") as f:
        f.write(report)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.savefig("results/confusion_matrix.png")
    print("Confusion Matrix saved to results/confusion_matrix.png")

if __name__ == "__main__":
    # Define paths and parameters
    data_dir = "data"  # Path to dataset
    model_path = "models/svm_model.pkl"  # Path to saved model
    img_size = 512  # Image size for resizing

    # Make sure results directory exists
    if not os.path.exists("results"):
        os.makedirs("results")

    # Load and preprocess data
    from preprocess import load_data
    from feature_extraction import extract_sift_features

    print("Loading data for evaluation...")
    X, y, classes = load_data(data_dir, img_size)
    X_features = extract_sift_features(X)

    # Split data for evaluation
    from sklearn.model_selection import train_test_split
    _, X_test, _, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    # Evaluate the model
    evaluate_model(model_path, X_test, y_test, classes)

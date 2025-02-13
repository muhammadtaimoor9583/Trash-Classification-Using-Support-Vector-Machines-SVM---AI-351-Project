import joblib
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, GridSearchCV
from preprocess import load_data
from feature_extraction import extract_sift_features

def train_svm_with_tuning(data_dir, img_size, model_path):
    # Load and preprocess data
    print("Loading and preprocessing data...")
    X, y, classes = load_data(data_dir, img_size)
    X_features = extract_sift_features(X)

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_features, y, test_size=0.2, random_state=42)

    # Perform GridSearchCV for hyperparameter tuning
    print("Tuning hyperparameters using GridSearchCV...")
    param_grid = {
        'C': [0.1, 0.5,1, 5, 10, 15, 20, 100],          # Test different values of C
        'gamma': ['scale', 0.1, 0.01]    # Test different values of gamma (optional)
    }
    grid_search = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)

    # Get the best model and parameters
    best_svm = grid_search.best_estimator_
    print("Best Parameters:", grid_search.best_params_)
    print("Best Cross-Validation Accuracy:", grid_search.best_score_)

    # Save the best model
    joblib.dump((best_svm, classes), model_path)
    print(f"Best model saved to: {model_path}")

    # Return data and classes for evaluation
    return X_test, y_test, classes

if __name__ == "__main__":
    # Define paths and parameters
    data_dir = "data"  # Path to the dataset
    model_path = "models/svm_model.pkl"  # Path to save the best model
    img_size = 512  # Image size for resizing

    # Create directories if they don't exist
    import os
    if not os.path.exists("models"):
        os.makedirs("models")

    # Train the SVM with GridSearchCV
    print("Training SVM with hyperparameter tuning...")
    X_test, y_test, classes = train_svm_with_tuning(data_dir, img_size, model_path)
    print("Training complete!")



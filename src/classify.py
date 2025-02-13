import joblib
import cv2
import os
import numpy as np
from feature_extraction import extract_sift_features

def classify_image(image_path, model_path):
    # Load the trained SVM model
    svm, classes = joblib.load(model_path)

    # Load and preprocess the image
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (512, 384))
    features = extract_sift_features([img])

    # Predict the class
    predicted_class = svm.predict(features)[0]
    print(f"Classified as: {classes[predicted_class]}")
    return classes[predicted_class]

if __name__ == "__main__":
    # Define paths
    image_path = "trash"
    model_path = "models/svm_model.pkl"  # Path to the saved model

    # Get all image files in the trash folder
    image_folder = "trash"
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if f.endswith(('.png', '.jpg', '.jpeg'))]

    # Classify the image
    if not os.path.exists(model_path):
        print("Model not found! Train the model first by running train_svm.py.")
    else:
        for image in image_files:
            print(image)
            classify_image(image, model_path)

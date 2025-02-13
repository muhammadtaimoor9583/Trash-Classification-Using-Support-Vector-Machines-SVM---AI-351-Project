import os
import cv2
import numpy as np



def load_data(data_dir, img_size):
    X = []  # Images
    y = []  # Labels
    classes = os.listdir(data_dir)  # Folder names = class names
  

    for label, folder in enumerate(classes):
        folder_path = os.path.join(data_dir, folder)
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # Load grayscale image
            img = cv2.resize(img, (img_size, 384))  # Resize to fixed size
            X.append(img)
            y.append(label)

    return np.array(X), np.array(y), classes

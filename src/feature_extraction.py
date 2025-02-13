import cv2
import numpy as np

def extract_sift_features(images):
    sift = cv2.SIFT_create()
    feature_vectors = []

    for img in images:
        keypoints, descriptors = sift.detectAndCompute(img, None)
        if descriptors is not None:
            # Average pooling to create a fixed-length vector
            feature_vectors.append(np.mean(descriptors, axis=0))
        else:
            # Handle images with no keypoints
            feature_vectors.append(np.zeros(128))

    return np.array(feature_vectors)

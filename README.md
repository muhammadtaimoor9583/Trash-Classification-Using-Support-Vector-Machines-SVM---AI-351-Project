# Trash Classification Using Support Vector Machines (SVM) - AI 351 Project

This repository contains an image classification project using Support Vector Machines (SVM). The project is part of the AI 351 course and focuses on classifying trash images into different categories.

## 📌 Table of Contents
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

## 📖 Introduction
Trash classification is an essential step in automating waste management. This project uses **Support Vector Machines (SVM)** to classify images of trash into different categories. 

### 🔹 Features:
- Image preprocessing and feature extraction
- SVM-based classification model
- Performance evaluation using accuracy, confusion matrix, and classification reports

## ⚙️ Installation
To set up the project, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone https://github.com/hash-walker/AI_351_SVM_image_classification.git
   ```
2. **Navigate to the project directory:**
   ```bash
   cd AI_351_SVM_image_classification
   ```
3. **Install the required dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🚀 Usage
To train the SVM model and classify images, run the following command:

```bash
python train.py
```

For more detailed usage instructions and configurable parameters, refer to the `train.py` script.

## 📂 Dataset
The dataset used for this project should be placed in the `data/` directory. Make sure that:
- The dataset is structured into labeled folders.
- It is split into **training** and **testing** sets.

## 📊 Results
The results of the image classification process will be stored in the `results/` directory. This includes:
- Model accuracy
- Confusion matrix
- Precision, recall, and F1-score
- Visualization of classified images

## 🤝 Contributing
Contributions are welcome! If you’d like to improve the project, follow these steps:
1. **Fork** the repository.
2. Create a new **feature branch**.
3. **Commit** your changes with descriptive messages.
4. Push your branch and **submit a pull request**.

## 📜 License
This project is licensed under the **MIT License**. Feel free to use and modify the code as needed.

---

**Author:** [Hash Walker](https://github.com/hash-walker)  
For any inquiries or contributions, feel free to reach out via GitHub.

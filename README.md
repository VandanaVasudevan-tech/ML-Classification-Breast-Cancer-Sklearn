# 🧠 Breast Cancer Classification using Supervised Machine Learning

## 📌 Project Overview

This project applies multiple **supervised machine learning classification algorithms** to the Breast Cancer dataset available in the Scikit-learn library.
The objective is to compare the performance of different models in predicting whether a tumor is **benign** or **malignant**.

This project was completed as part of a Machine Learning assessment.

---

## 🎯 Objective

To evaluate understanding of:

* Data preprocessing
* Feature scaling
* Implementation of classification algorithms
* Model evaluation and comparison

---

## 📊 Dataset Information

The dataset is loaded from **sklearn.datasets.load_breast_cancer()**

**Dataset Characteristics**

* 569 samples
* 30 numerical features
* Binary classification:

  * 0 → Malignant (Cancerous)
  * 1 → Benign (Non-Cancerous)

---

## ⚙️ Preprocessing Steps

The following preprocessing steps were performed:

1. Converted dataset into **Pandas DataFrame and Series**
2. Checked for missing values (none found)
3. Applied **StandardScaler** for feature scaling

Feature scaling is important because algorithms like Logistic Regression, SVM and KNN rely on distance calculations and gradient-based optimization.

---

## 🤖 Machine Learning Models Used

The following five classification algorithms were implemented:

1️⃣ Logistic Regression
2️⃣ Decision Tree Classifier
3️⃣ Random Forest Classifier
4️⃣ Support Vector Machine (SVM)
5️⃣ K-Nearest Neighbors (KNN)

---

## 📈 Evaluation Metrics

Models were evaluated using:

* Accuracy Score
* Confusion Matrix
* Classification Report (Precision, Recall, F1-Score)

---

## 🏆 Model Comparison Result

| Model                  | Performance           |
| ---------------------- | --------------------- |
| Logistic Regression    | High Accuracy         |
| Decision Tree          | Slight Overfitting    |
| Random Forest          | Excellent Performance |
| Support Vector Machine | Best Performance      |
| K-Nearest Neighbors    | Very Good Performance |

### ✅ Best Performing Models

**Support Vector Machine & Random Forest**

### ❌ Lowest Performing Model

**Decision Tree Classifier**

---

## 📌 Conclusion

This project demonstrates how different supervised learning algorithms perform on a medical dataset.
The results show that **ensemble models and margin-based classifiers** provide the most reliable predictions for breast cancer diagnosis.

---

## 🛠️ Technologies Used

* Python
* Scikit-learn
* Pandas
---



**Vandana Vasudevan**
Python Developer | Aspiring Data Scientist

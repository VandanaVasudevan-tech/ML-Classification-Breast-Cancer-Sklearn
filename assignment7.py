import numpy as np
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
# ======================================================================================================================
# STEP 1: Load the Dataset
# The Breast Cancer dataset contains medical features computed from breast mass images.
# The goal is to classify tumors as:
# 0 → Malignant (Cancerous)
# 1 → Benign (Non-cancerous)
# Dataset contains:
# 569 samples
# 30 numerical features
# This is a binary classification problem.
# ======================================================================================================================
print("Step 1: Loading Breast Cancer Dataset")
data = load_breast_cancer()
print("Dataset loaded successfully")
# ======================================================================================================================
# STEP 2: DATAFRAME CREATION
# Features are converted into a pandas DataFrame to
# improve readability and allow easy data manipulation.
# The target variable is kept separate to avoid data leakage.
# ======================================================================================================================
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name="target")
print(data.target_names)
print("Feature shape:", X.shape)
print("Target shape:", y.shape)
# ======================================================================================================================
# STEP 3: Combine Features and Target (For Visualization)
# ======================================================================================================================
print("\nStep 3: Creating combined DataFrame for visualization")
df = pd.concat([X, y], axis=1)
print(df.head())
# ======================================================================================================================
# STEP 4: Check for Missing Values
# Checking for missing values is a mandatory preprocessing
# step to ensure data quality. Missing values can negatively
# impact model performance and cause errors during training.
# ======================================================================================================================
print("\nStep 4: Checking for missing values")
print(X.isnull().sum())
# ======================================================================================================================
# STEP 5: Feature Scaling
# The dataset contains features with different magnitudes
# Standardization is applied
# to bring all features to a common scale with mean 0 and
# standard deviation 1.
# Algorithms like Logistic Regression, SVM, KNN depend on distance calculations.
# Scaling improves accuracy and convergence speed.
# ======================================================================================================================
print("\nStep 5: Applying Standardization")
scaler = StandardScaler()
scaled_data = scaler.fit_transform(X)
X_scaled = pd.DataFrame(scaled_data, columns=X.columns)
print("Feature scaling completed")
print(X_scaled.head())

print("Interpretation:\n"
      "The preprocessing steps involved converting the breast cancer dataset into pandas DataFrame and Series "
      "format to enable easier exploration and manipulation of the data. The dataset was checked for missing "
      "values to ensure data quality and completeness, and no missing values were found. Feature scaling using "
      "StandardScaler was applied because the dataset contains features with different numerical ranges. "
      "Scaling is particularly important for classification algorithms such as Logistic Regression, Support "
      "Vector Machine, and K-Nearest Neighbors, which rely on distance calculations and gradient-based "
      "optimization. Standardization ensures that all features contribute equally to the model, improves "
      "training stability, and helps achieve better classification performance.")
# ======================================================================================================================
# STEP 1: TRAIN-TEST SPLIT
# The dataset is split into training and testing sets
# to evaluate the model's performance on unseen data.
# ======================================================================================================================
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
print("Train-test split completed")

# -----------------------------Logistic Regression----------------------------------------------------------------------
# ======================================================================================================================
# STEP 1: MODEL CREATION AND TRAINING
# Logistic Regression predicts the probability of a class using
# the sigmoid function and finds the best decision boundary.

# WHY SUITABLE :
# Breast cancer dataset is binary classification and mostly linearly separable.
# Logistic Regression works well as a strong baseline classifier.
# ======================================================================================================================
log_model = LogisticRegression(max_iter=5000)
log_model.fit(X_train, y_train)
print("Logistic Regression model trained successfully")
# ======================================================================================================================
# STEP 2: MAKE PREDICTION ON TEST DATA
# ======================================================================================================================
log_pred = log_model.predict(X_test)
# ======================================================================================================================
# STEP 3: MODEL EVALUATION
# Accuracy, Confusion Matrix and Classification Report are used
# to evaluate classification performance.
# ======================================================================================================================
log_acc = accuracy_score(y_test, log_pred)
print("Logistic Regression Performance:")
print("Accuracy:", log_acc)
print(confusion_matrix(y_test, log_pred))
print(classification_report(y_test, log_pred))
# ======================================================================================================================
# --------------------------------------DECISION TREE CLASSIFIER--------------------------------------------------------
# ======================================================================================================================
# STEP 1: MODEL CREATION AND TRAINING
# Decision Tree splits the dataset based on feature thresholds
# to create a tree-like structure for classification.

# WHY SUITABLE :
# Captures non-linear patterns and feature interactions.
# Easy to interpret for medical decision making.
# ======================================================================================================================
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train, y_train)
print("Decision Tree model trained successfully")
dt_pred = dt_model.predict(X_test)
dt_acc = accuracy_score(y_test, dt_pred)
print("Decision Tree Performance:")
print("Accuracy:", dt_acc)
print(confusion_matrix(y_test, dt_pred))
print(classification_report(y_test, dt_pred))
# -----------------------------------------RANDOM FOREST CLASSIFIER-----------------------------------------------------
# ======================================================================================================================
# Random Forest builds multiple decision trees and combines their predictions.
# This reduces overfitting and improves accuracy.

# WHY SUITABLE :
# Medical datasets contain complex patterns and feature interactions.
# Random Forest provides robust and stable predictions.
# ======================================================================================================================
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train, y_train)
print("Random Forest model trained successfully")

rf_pred = rf_model.predict(X_test)
rf_acc = accuracy_score(y_test, rf_pred)

print("Random Forest Performance:")
print("Accuracy:", rf_acc)
print(confusion_matrix(y_test, rf_pred))
print(classification_report(y_test, rf_pred))

# -----------------------------------------SUPPORT VECTOR MACHINE-------------------------------------------------------
# ======================================================================================================================
# SVM finds the optimal hyperplane that maximizes the margin
# between two classes.

# WHY SUITABLE :
# Works extremely well for high-dimensional datasets
# like the breast cancer dataset.
# ======================================================================================================================
svm_model = SVC()
svm_model.fit(X_train, y_train)
print("SVM model trained successfully")

svm_pred = svm_model.predict(X_test)
svm_acc = accuracy_score(y_test, svm_pred)

print("SVM Performance:")
print("Accuracy:", svm_acc)
print(confusion_matrix(y_test, svm_pred))
print(classification_report(y_test, svm_pred))
# -----------------------------------------K-NEAREST NEIGHBORS (KNN)----------------------------------------------------
# ======================================================================================================================
# KNN classifies data based on the majority class of nearest neighbors.

# WHY SUITABLE :
# Works well for smaller datasets and performs well after scaling.
# Uses distance-based learning.
# ======================================================================================================================
knn_model = KNeighborsClassifier(n_neighbors=5)
knn_model.fit(X_train, y_train)
print("KNN model trained successfully")

knn_pred = knn_model.predict(X_test)
knn_acc = accuracy_score(y_test, knn_pred)

print("KNN Performance:")
print("Accuracy:", knn_acc)
print(confusion_matrix(y_test, knn_pred))
print(classification_report(y_test, knn_pred))
# ======================================================================================================================
print("MODEL COMPARISON")
print("Logistic Regression Accuracy:", log_acc)
print("Decision Tree Accuracy:", dt_acc)
print("Random Forest Accuracy:", rf_acc)
print("SVM Accuracy:", svm_acc)
print("KNN Accuracy:", knn_acc)
print("""FINAL INTERPRETATION:
After comparing all five classification algorithms, SVM and Random Forest achieved the highest accuracy,
showing excellent performance in classifying tumors as benign or malignant.
Decision Tree showed comparatively lower performance due to overfitting.
Therefore, ensemble and margin-based models are the most suitable for the breast cancer dataset.""")
# ======================================================================================================================
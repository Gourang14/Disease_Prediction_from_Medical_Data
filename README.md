# Disease Prediction from Medical Data
This repository contains the code for building a disease prediction model using multiple machine learning algorithms and a neural network model.

# Overview
The purpose of this project is to develop a model that can predict the likelihood of a disease based on patient symptoms and profiles. The dataset includes various health indicators such as age, gender, fever, cough, fatigue, and other vital signs, which are then used to build and evaluate predictive models.

# Dataset
The dataset used in this project is the "https://www.kaggle.com/datasets/uom190346a/disease-symptoms-and-patient-profile-dataset" which contains patient information such as symptoms and disease outcomes. You can find the dataset here. The dataset consists of the following columns:

Age: Age of the patient.
Gender: Gender of the patient (Male/Female).
Fever: Whether the patient has a fever (Yes/No).
Cough: Whether the patient has a cough (Yes/No).
Fatigue: Whether the patient experiences fatigue (Yes/No).
Difficulty Breathing: Whether the patient experiences difficulty breathing (Yes/No).
Blood Pressure: The blood pressure level of the patient.
Cholesterol Level: Cholesterol level of the patient.
Disease: The specific disease diagnosed in the patient.
Results: The outcome variable indicating whether the patient is diagnosed with the disease (Yes/No).

# Data Cleaning & Preprocessing
The dataset goes through multiple preprocessing steps, including:

Handling Missing Values: Missing values in the "Age" column are filled with the mean age.
Duplicate Removal: Duplicate rows are removed to ensure data integrity.
Encoding: The categorical variables are encoded using OneHotEncoder and LabelEncoder.
Scaling: Feature values are scaled using MinMaxScaler.

# Exploratory Data Analysis (EDA)
To gain insights into the data, the following visualizations are generated:

Pie Charts: Distribution of the results (disease diagnosis) based on age groups (less than 45 and more than 45 years old).
# Models Implemented
Logistic Regression:

L1-penalty based Logistic Regression is applied.
Decision Tree:

A Decision Tree model with a max depth of 6 is used.
K-Nearest Neighbors (KNN):

A KNN classifier with default hyperparameters is applied.
Neural Network:

A Sequential Neural Network model with two hidden layers is built using Keras. The model uses binary_crossentropy loss and adam optimizer.

# Model Evaluation
The models are evaluated based on several metrics, including:

Accuracy
Precision
Recall
F1 Score
Confusion Matrix
ROC-AUC Score
A comprehensive evaluation function evaluate_model() prints these metrics for each model.

# Hyperparameter Tuning
Hyperparameter tuning is conducted using GridSearchCV for the following models:

K-Nearest Neighbors: Tuning the number of neighbors and distance metrics.
Decision Tree: Tuning the tree depth and minimum samples required for a split.
Best parameters found during tuning:

KNN: {'metric': 'euclidean', 'n_neighbors': 7}
Decision Tree: {'max_depth': 10, 'min_samples_split': 2}
Feature Importance
Feature importance is visualized for the Decision Tree model, allowing us to identify the most significant features that influence disease prediction.

# ROC Curve Visualization
The Receiver Operating Characteristic (ROC) curves are plotted to evaluate the true positive rate (TPR) versus false positive rate (FPR) for different models.

# Model Persistence
The trained models and scaler are saved for future use with joblib:

Model: disease_prediction_model.joblib
Scaler: min_max_scaler.joblib
# Predictions
The model is capable of making predictions for both individual patients and multiple patients at once. The function load_model_and_predict() loads the saved model and scaler to perform predictions on new patient data.

# Conclusion
This project successfully demonstrates the development of a disease prediction model using various machine learning algorithms such as Logistic Regression, Decision Tree, K-Nearest Neighbors, and a Neural Network. The model's performance was evaluated using metrics like accuracy, precision, recall, F1 score, and ROC curve analysis. The neural network showed promising results in terms of accuracy and predictive capability. Additionally, hyperparameter tuning helped improve the models' performance. This predictive model can serve as a valuable tool for healthcare professionals in early disease detection and diagnosis based on patient symptoms and profiles, ultimately contributing to more efficient and timely medical interventions.

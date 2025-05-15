# 🎓 Student Performance Classifier: Pass/Fail Prediction using MLP and SVM

This project applies machine learning techniques to predict whether a student will pass or fail based on various personal, academic, and social features. The data is sourced from a real-world dataset on student achievement in secondary education.

## 📁 Project Structure

- `student-mat.csv` — The dataset containing student information and final grades.
- `PassFail Classifier.py` — Python script that processes the data, trains two classifiers (MLP and SVM), evaluates their performance, and visualizes the results.

## ⚙️ Technologies Used

- Python
- Scikit-learn
- Pandas, NumPy
- Matplotlib

## 🧠 Models Implemented

- **Multi-Layer Perceptron (MLP)**: A feedforward neural network with 2 hidden layers, trained incrementally over 5 epochs using K-Fold cross-validation.
- **Support Vector Machine (SVM)**: Trained with a radial basis function (RBF) kernel, evaluated using 10-fold cross-validation.

## 🧪 Evaluation Metrics

Each model is evaluated using:
- Accuracy
- Precision
- Recall
- F1-Score
- ROC AUC
- Confusion Matrix
- Train/Validation Error Plots
- ROC Curve Comparison

## 📊 Visual Outputs

The script generates:
- ROC curves for both models
- Training and validation error plots across epochs (MLP)
- Training and validation error per fold (SVM)

## 🏁 Final Results Summary

At the end of the script, the performance metrics are compared, and the best-performing model is reported.

## 🚀 Getting Started

### Prerequisites

Install required libraries:
```bash
pip install pandas numpy scikit-learn matplotlib

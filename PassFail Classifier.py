import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, precision_score, recall_score, f1_score
import matplotlib.pyplot as plt

file_path = "student-mat.csv"
data = pd.read_csv(file_path, sep=';')


# Preprocess data
def preprocess_data(data):
    data = pd.get_dummies(data, drop_first=True)
    X = data.drop("G3", axis=1)
    y = data["G3"]
    y = (y >= 10).astype(int)
    return X, y


X, y = preprocess_data(data)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# K-Fold Cross Validation
kf = KFold(n_splits=10, shuffle=True)

# --- 1. Neural Network Model (MLPClassifier) ---
mlp_model = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', max_iter=1,
                           tol=1e-4)

# Train Neural Network with K-Fold and record error for each fold
train_errors_mlp = []
test_errors_mlp = []

for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    epoch_train_errors = []
    epoch_val_errors = []


    for epoch in range(1, 6):  # Training for 5 epochs
        mlp_model.partial_fit(X_tr, y_tr, classes=np.unique(y_train))  # Incremental fitting

        # Compute training and validation errors
        train_error = 1 - mlp_model.score(X_tr, y_tr)
        val_error = 1 - mlp_model.score(X_val, y_val)

        epoch_train_errors.append(train_error * 100)
        epoch_val_errors.append(val_error * 100)

        print(f"Fold: {len(train_errors_mlp) + 1}, Epoch {epoch} -> "
              f"Train Error: {train_error * 100:.2f}%, Validation Error: {val_error * 100:.2f}%")

    train_errors_mlp.append(epoch_train_errors)
    test_errors_mlp.append(epoch_val_errors)
# Calculate the mean errors for MLP over all folds
mean_train_errors_mlp = np.mean(train_errors_mlp, axis=0)
mean_test_errors_mlp = np.mean(test_errors_mlp, axis=0)

# Final evaluation on test set for Neural Network
y_pred_mlp = mlp_model.predict(X_test)
mlp_test_acc = accuracy_score(y_test, y_pred_mlp)
print("\n--- Neural Network (MLP) Results ---")
print("Neural Network Test Error:", 1 - mlp_test_acc)

# MLP Confusion Matrix
mlp_conf_matrix = confusion_matrix(y_test, y_pred_mlp)
tn_mlp, fp_mlp, fn_mlp, tp_mlp = mlp_conf_matrix.ravel()

print("Confusion Matrix (Neural Network):\n", mlp_conf_matrix)
print(f"True Negative (MLP): {tn_mlp}, False Positive (MLP): {fp_mlp}")
print(f"False Negative (MLP): {fn_mlp}, True Positive (MLP): {tp_mlp}")
# MLP Performance Metrics
mlp_accuracy = accuracy_score(y_test, y_pred_mlp)
mlp_precision = precision_score(y_test, y_pred_mlp)
mlp_recall = recall_score(y_test, y_pred_mlp)
mlp_f1 = f1_score(y_test, y_pred_mlp)
print(f"Accuracy: {mlp_accuracy:.4f}")
print(f"Precision: {mlp_precision:.4f}")
print(f"Recall: {mlp_recall:.4f}")
print(f"F1-Score: {mlp_f1:.4f}\n")

# --- 2. SVM Model ---
svm_model = SVC(kernel='rbf', C=1.0, probability=True)

# Train SVM Model with K-Fold and record error for each fold
train_errors_svm = []
test_errors_svm = []

for train_index, val_index in kf.split(X_train):
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[val_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[val_index]

    svm_model.fit(X_tr, y_tr)

    # Store the errors for each epoch

    # Calculate errors for each epoch
    train_error = 1 - accuracy_score(y_tr, svm_model.predict(X_tr))
    val_error = 1 - accuracy_score(y_val, svm_model.predict(X_val))

    train_errors_svm.append(train_error * 100)
    test_errors_svm.append(val_error * 100)
    print(f" Train Error = {train_error * 100:.2f}%, Validation Error = {val_error * 100:.2f}%")

    # Append errors for each fold


# Calculate the mean errors for SVM over all folds
mean_train_errors_svm = np.mean(train_errors_svm, axis=0)
mean_test_errors_svm = np.mean(test_errors_svm, axis=0)

# Final evaluation on test set for SVM
svm_model.fit(X_train, y_train)  # Refit on the entire training set
y_pred_svm = svm_model.predict(X_test)
svm_test_acc = accuracy_score(y_test, y_pred_svm)
print("\n--- SVM Results ---")
print(f"SVM Test Error: {1 - svm_test_acc:.2f}")
# SVM Confusion Matrix
svm_conf_matrix = confusion_matrix(y_test, y_pred_svm)
tn_svm, fp_svm, fn_svm, tp_svm = svm_conf_matrix.ravel()

print("Confusion Matrix (SVM):\n", svm_conf_matrix)
print(f"True Negative (SVM): {tn_svm}, False Positive (SVM): {fp_svm}")
print(f"False Negative (SVM): {fn_svm}, True Positive (SVM): {tp_svm}")

# SVM Performance Metrics
svm_accuracy = accuracy_score(y_test, y_pred_svm)
svm_precision = precision_score(y_test, y_pred_svm)
svm_recall = recall_score(y_test, y_pred_svm)
svm_f1 = f1_score(y_test, y_pred_svm)
print(f"Accuracy: {svm_accuracy:.4f}")
print(f"Precision: {svm_precision:.4f}")
print(f"Recall: {svm_recall:.4f}")
print(f"F1-Score: {svm_f1:.4f}\n")
# --- Plot ROC Curve for Both Models ---
# MLP ROC
mlp_probs = mlp_model.predict_proba(X_test)[:, 1]
fpr_mlp, tpr_mlp, _ = roc_curve(y_test, mlp_probs)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)

# SVM ROC
svm_probs = svm_model.predict_proba(X_test)[:, 1]
fpr_svm, tpr_svm, _ = roc_curve(y_test, svm_probs)
roc_auc_svm = auc(fpr_svm, tpr_svm)

# Plot ROC Curves for both models
plt.figure(figsize=(10, 6))
plt.plot(fpr_mlp, tpr_mlp, label=f'Neural Network (AUC = {roc_auc_mlp:.2f})')
plt.plot(fpr_svm, tpr_svm, label=f'SVM (AUC = {roc_auc_svm:.2f})')
plt.plot([0, 1], [0, 1], 'k--')  # Random guess line
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# Plot Error over Epochs for MLP
plt.figure(figsize=(10, 6))
plt.plot(range(1, 6), mean_train_errors_mlp, label="Training Error (MLP)")
plt.plot(range(1, 6), mean_test_errors_mlp, label="Validation Error (MLP)")
plt.xlabel('Epoch')
plt.ylabel('Error (%)')
plt.title('Error over Epochs (MLP)')
plt.legend()
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(range(1, len(train_errors_svm) + 1), train_errors_svm, label="Training Error (SVM)", marker='o')
plt.plot(range(1, len(test_errors_svm) + 1), test_errors_svm, label="Validation Error (SVM)", marker='o')
plt.xticks(range(1, len(train_errors_svm) + 1))
plt.xlabel('Fold')
plt.ylabel('Error (%)')
plt.title('Train and Validation Errors Across Folds (SVM)')
plt.legend()
plt.grid()
plt.show()

print("\n--- Model Performance Summary ---")
print(f"Neural Network Accuracy: {mlp_accuracy:.2f}")
print(f"Neural Network AUC: {roc_auc_mlp:.2f}")
print(f"SVM Accuracy: {svm_accuracy:.2f}")
print(f"SVM AUC: {roc_auc_svm:.2f}")

if mlp_accuracy > svm_accuracy:
    print("\n--- Best Model ---\nThe best model is the Neural Network (MLP).")
else:
    print("\n--- Best Model ---\nThe best model is the SVM.")

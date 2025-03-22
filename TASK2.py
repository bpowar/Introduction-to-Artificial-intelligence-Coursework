import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix, classification_report

#function to make feature names unique- implemented this because it gave me an error about repeated names 
def make_unique(names):
    seen = {}
    unique_names = []
    for name in names:
        if name in seen:
            seen[name] += 1
            unique_names.append(f"{name}_{seen[name]}")
        else:
            seen[name] = 0
            unique_names.append(name)
    return unique_names

# 1. Load and Preprocess the HAR Data
BASE_DIR = r"C:\Users\USER\Desktop\AI CW\UCI HAR Dataset"

#define file paths
features_fp = os.path.join(BASE_DIR, "features.txt")
activities_fp = os.path.join(BASE_DIR, "activity_labels.txt")
train_X_fp = os.path.join(BASE_DIR, "train", "X_train.txt")
train_y_fp = os.path.join(BASE_DIR, "train", "y_train.txt")
test_X_fp = os.path.join(BASE_DIR, "test", "X_test.txt")
test_y_fp = os.path.join(BASE_DIR, "test", "y_test.txt")

#load feature names
features_df = pd.read_csv(features_fp, delim_whitespace=True, header=None, names=["Index", "Feature"])
feature_names = features_df["Feature"].tolist()
feature_names = make_unique(feature_names)  # Ensure unique column names

#load activity labels and create a mapping dictionary
activities_df = pd.read_csv(activities_fp, delim_whitespace=True, header=None, names=["ID", "Activity"])
activity_map = dict(zip(activities_df["ID"], activities_df["Activity"]))

#read training+ test data using the unique names
X_train = pd.read_csv(train_X_fp, delim_whitespace=True, header=None, names=feature_names)
y_train = pd.read_csv(train_y_fp, delim_whitespace=True, header=None, names=["Activity"])
X_test = pd.read_csv(test_X_fp, delim_whitespace=True, header=None, names=feature_names)
y_test = pd.read_csv(test_y_fp, delim_whitespace=True, header=None, names=["Activity"])

#map numeric activity codes to their string labels
y_train["Activity"] = y_train["Activity"].map(activity_map)
y_test["Activity"] = y_test["Activity"].map(activity_map)

# 2. Convert the Original 6-Class Labels into a Binary Problem
#define binary labeling:
#   Active (1): WALKING, WALKING_UPSTAIRS, WALKING_DOWNSTAIRS
#   Inactive (0): SITTING, STANDING, LAYING
def to_binary_label(activity):
    return 1 if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"] else 0

#apply conversion on both training and test sets
y_train["Binary"] = y_train["Activity"].apply(to_binary_label)
y_test["Binary"] = y_test["Activity"].apply(to_binary_label)

# 3. Train Baseline SVM Models with Different Kernels
baseline_models = {}
kernels_to_try = ['linear', 'poly', 'rbf']

print("=== Baseline SVM Models ===")
for kernel in kernels_to_try:
    print(f"\n-- Training SVM with kernel: {kernel} --")
    svm_model = SVC(kernel=kernel, random_state=42)
    svm_model.fit(X_train, y_train["Binary"])
    preds = svm_model.predict(X_test)
    
    #evaluate model performance using confusion matrix + classification report
    cm = confusion_matrix(y_test["Binary"], preds)
    print("Confusion Matrix:")
    print(cm)
    print("Classification Report:")
    print(classification_report(y_test["Binary"], preds))
    
    baseline_models[kernel] = svm_model

# 4. Hyperparameter Tuning using GridSearchCV
# build a pipeline that scales features 
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    # ('pca', PCA(n_components=50)),  # reduced features
    ('svm', SVC())
])

#define hyperparameter grid for different kernels and their parameters
param_grid = [
    {
        'svm__kernel': ['linear'],
        'svm__C': [0.1, 1, 10, 100]
    },
    {
        'svm__kernel': ['poly'],
        'svm__C': [0.1, 1],
        'svm__degree': [2, 3],
        'svm__gamma': [0.001, 0.01, 0.1]
    },
    {
        'svm__kernel': ['rbf'],
        'svm__C': [0.1, 1, 10],
        'svm__gamma': [0.001, 0.01, 0.1]
    }
]

grid_search = GridSearchCV(estimator=pipeline, 
                           param_grid=param_grid,
                           cv=3,
                           scoring='accuracy',
                           n_jobs=-1,
                           verbose=1)

grid_search.fit(X_train, y_train["Binary"].values.ravel())

print("\n=== GridSearchCV Results ===")
print("Best Parameters:", grid_search.best_params_)
print("Best Cross-Validation Accuracy:", grid_search.best_score_)

# 5. Evaluate and Interpret Results
best_svm = grid_search.best_estimator_
y_pred_best = best_svm.predict(X_test)

final_cm = confusion_matrix(y_test["Binary"], y_pred_best)
final_report = classification_report(y_test["Binary"], y_pred_best)

print("\n--- Evaluation of the Tuned SVM Model ---")
print("Confusion Matrix:")
print(final_cm)
print("\nClassification Report:")
print(final_report)

plt.figure(figsize=(6, 5))
sns.heatmap(final_cm, annot=True, fmt='d', cmap='viridis',
            xticklabels=['Inactive (0)', 'Active (1)'],
            yticklabels=['Inactive (0)', 'Active (1)'])
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix for Tuned SVM Model")
plt.show()

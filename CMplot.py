import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix
from joblib import load

# Load the trained models and predictions
best_clf_rf = load("datafiles/results/Random Forest/random_forest_model.joblib")
y_pred_rf = load("datafiles/results/Random Forest/random_forest_predictions.joblib")
y_test_rf = load("datafiles/results/Random Forest/random_forest_tests.joblib")
best_clf_svm = load("datafiles/results/Support Vector Machine/svm_model.joblib")
y_pred_svm = load("datafiles/results/Support Vector Machine/svm_predictions.joblib")
y_test_svm = load("datafiles/results/Support Vector Machine/svm_tests.joblib")
best_clf_knn = load("datafiles/results/K-Nearest Neighbors/knn_model.joblib")
y_pred_knn = load("datafiles/results/K-Nearest Neighbors/knn_predictions.joblib")
y_test_knn = load("datafiles/results/K-Nearest Neighbors/knn_tests.joblib")

# Define class labels
class_labels = best_clf_rf.classes_

# Assuming all models have the same class labels
models = ["Random Forest", "Support Vector Machine", "K-Nearest Neighbors"]
confusion_matrices = []

# Calculate confusion matrices for each model
y_tests = [y_test_rf, y_test_svm, y_test_knn]
y_preds = [y_pred_rf, y_pred_svm, y_pred_knn]
for y_test, y_pred in zip(y_tests, y_preds):
    cm = confusion_matrix(y_test, y_pred, labels=class_labels)
    confusion_matrices.append(cm)

# Plotting the confusion matrices using seaborn
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for i, (cm, model) in enumerate(zip(confusion_matrices, models)):
    cm_percentage = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] * 100
    cm_df = pd.DataFrame(cm_percentage, index=class_labels, columns=class_labels)
    cm_percentage_df = pd.DataFrame(
        cm_percentage, index=class_labels, columns=class_labels
    )

    # Check if the count is 0
    group_counts = ["{0:0.0f}".format(value) if value != 0 else "" for value in cm.flatten()]

    # Check if the percentage is 0
    group_percentages = ["{0:.0f}%".format(value) if value != 0 else "" for value in cm_percentage.flatten()]

    # Combine the count and percentage, but use an empty string if both are 0
    labels = [f"{v1}\n{v2}" if v1 != "" or v2 != "" else "" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(cm.shape)

    sns_heatmap = sns.heatmap(
        cm_percentage_df,
        annot=labels,
        fmt="",
        cmap="Blues",
        square=True,
        annot_kws={"size": 14},
        vmin=0,
        vmax=100,
        cbar=False,
        ax=axes[i],
    )
    axes[i].set_title(model, pad=20, fontsize=18)
    if i == 0:
        axes[i].set_ylabel("True Building Class", labelpad=20, fontsize=14)
        axes[i].set_yticklabels(axes[i].get_yticklabels(), rotation=0, fontsize=12)
    else:
        axes[i].set_ylabel("")
        axes[i].set_yticklabels(
            []
        )  # Set y-tick labels to an empty list for subplots 2 and 3

    axes[i].set_xticklabels(
        axes[i].get_xticklabels(), fontsize=12
    )  # Set x ticks rotation and fontsize

    if i == 1:
        axes[i].set_xlabel("Predicted Building Class", labelpad=20, fontsize=14)
    else:
        axes[i].set_xlabel("")

plt.tight_layout()
plt.show()

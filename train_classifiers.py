import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

#Change to the directory where the script is located
os.chdir(os.path.dirname(os.path.abspath(__file__)))

#Load the train/test data
X_train = pd.read_parquet("data/X_train.parquet")
X_test  = pd.read_parquet("data/X_test.parquet")
y_train = pd.read_parquet("data/y_train.parquet").iloc[:, 0]
y_test  = pd.read_parquet("data/y_test.parquet").iloc[:, 0]

#run a dummy classifier to get baseline metrics
from sklearn.dummy import DummyClassifier
from sklearn.metrics import classification_report, roc_auc_score, average_precision_score

#print baseline metrics
dummy = DummyClassifier(strategy="most_frequent").fit(X_train, y_train)
print("\nBaseline Metrics (Dummy Classifier):")
print(classification_report(y_test, dummy.predict(X_test), zero_division=0))

# Logistic Regression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)

# Define linear regression pipeline
lr_pipe = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000, C=1.0, random_state=42)),
])
lr_pipe.fit(X_train, y_train)

# Predictions
y_pred = lr_pipe.predict(X_test)
y_prob = lr_pipe.predict_proba(X_test)[:, 1]  # probability of positive class

#LR Metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_prob)
pr_auc = average_precision_score(y_test, y_prob)  # PR-AUC

print("\nLogistic Regression Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

#random forest classifier
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=None,
    min_samples_leaf=5,
    n_jobs=-1,
    random_state=42,
).fit(X_train, y_train)

#RF Metrics
accuracy = accuracy_score(y_test, rf.predict(X_test))
precision = precision_score(y_test, rf.predict(X_test), zero_division=0)
recall = recall_score(y_test, rf.predict(X_test), zero_division=0)
f1 = f1_score(y_test, rf.predict(X_test), zero_division=0)
roc_auc = roc_auc_score(y_test, rf.predict_proba(X_test)[:, 1])
pr_auc = average_precision_score(y_test, rf.predict_proba(X_test)[:, 1])  # PR-AUC

print("\nRandom Forest Classifier Metrics:")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1:        {f1:.4f}")
print(f"ROC-AUC:   {roc_auc:.4f}")
print(f"PR-AUC:    {pr_auc:.4f}")

#Cross-validation for both models
from sklearn.model_selection import cross_val_score

lr_cv = cross_val_score(lr_pipe, X_train, y_train, cv=5, scoring="average_precision")
rf_cv = cross_val_score(rf, X_train, y_train, cv=5, scoring="average_precision")

print("\nCross-Validation PR-AUC Scores:")
print(f"LR PR-AUC: {lr_cv.mean():.3f} ± {lr_cv.std():.3f}")
print(f"RF PR-AUC: {rf_cv.mean():.3f} ± {rf_cv.std():.3f}")

#Feature Importance for RF
importance = pd.Series(rf.feature_importances_, index=X_train.columns).sort_values(ascending=False)
importance.head(15).plot(kind="barh")
plt.title("RF feature importance — churn")
plt.savefig("figures/rf_importance.png", bbox_inches="tight")

#Feature Importance for LR (standardized coefficients)
print("\nLogistic Regression Coefficients (Standardized):")
coefs = pd.Series(lr_pipe.named_steps["clf"].coef_[0], index=X_train.columns).sort_values(key=abs, ascending=False)
print(coefs.head(15))

#take the top 400 predicted churners from the RF model and compute precision and recall
probs = rf.predict_proba(X_test)[:, 1]
# Sort descending, take the top 400 (~7% of the ~6,000-row test set)
sorted_probs = np.sort(probs)[::-1]
threshold = sorted_probs[399]   # the 400th highest prob
print(f"Threshold: {threshold:.3f}")

predictions_topk = (probs >= threshold).astype(int)
# Compute precision and recall at this threshold
precision_topk = precision_score(y_test, predictions_topk, zero_division=0)
recall_topk = recall_score(y_test, predictions_topk, zero_division=0)

print("\nTop-400 RF Predictions Metrics:")
print(f"Precision @ top 400: {precision_topk:.3f}")
print(f"Recall @ top 400:    {recall_topk:.3f}")

"""
model is barely better than the dummy baseline where it matters (finding churners), 
and in some ways it’s functionally unusable depending on the business goal.
If you had to ship something, it could be used as a ranked targeting system, not a classifier:
Score users by churn probability
Target top N (like your top-400 slice)
Frame it as: “Prioritized churn risk list” not “churn prediction”
"""
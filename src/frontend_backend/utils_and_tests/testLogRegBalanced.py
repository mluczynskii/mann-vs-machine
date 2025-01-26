import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import pickle
from src.models.logreg import LogisticRegressionModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score



df = pd.read_excel("src/frontend_backend/datasetBalanced.xlsx")

X = df.iloc[:, 0].values  # Tweets
y = df.iloc[:, 1].values  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

model = LogisticRegressionModel()
X_train_matrix = model.preprocess_training_set(X_train, custom=True)
X_test_matrix = model.preprocess_test_set(X_test, custom=True)
model.train(X_train_matrix, y_train, n_iters=300, learning_rate=0.001, regularization=None, alpha=1)

y_pred = model.classify(X_test_matrix)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, model.sigmoid(X_test_matrix))

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"ROC AUC: {roc_auc:.4f}")
print(f"Specifity: {specificity: 0.4f}")

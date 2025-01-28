import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import train_test_split
from src.models.tokenization import tokenize
from src.models.exception import NotTrainedError

from src.models.naive_bayes import NaiveBayesModel

df = pd.read_excel("src/frontend_backend/dataset.xlsx")

X = df.iloc[:, 0].values  # Tweets
y = df.iloc[:, 1].values  # Labels

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

nb_model = NaiveBayesModel()

X_train_tokenized = nb_model.preprocess_set(X_train)
X_test_tokenized = nb_model.preprocess_set(X_test)

nb_model.train(X_train_tokenized, y_train)

y_pred = nb_model.classify(X_test_tokenized)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
specificity = tn / (tn + fp)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")
print(f"Specificity: {specificity:.4f}")

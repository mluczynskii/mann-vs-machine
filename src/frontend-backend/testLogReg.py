import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import confusion_matrix
import pickle
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


def custom_tokenize(x):
    return str.casefold("".join(
        filter(lambda s: str.isalpha(s) or str.isspace(s),
               x))).split()


def gradient(xs, ys, beta):
    k = beta.shape[0]
    res = []
    aux = ys - np.reciprocal(1 + np.exp(-xs @ beta))
    for j in range(k):
        partial = np.dot(xs[:, j], aux)
        res.append(partial)
    return np.array(res)


def reg_gradient_descent(xs, ys, n_iters, learning_rate,
                         regularization=None, alpha=1):
    k = xs.shape[1]
    rng = np.random.default_rng()
    beta = rng.standard_normal(size=k)
    for _ in range(n_iters):
        if regularization == "l1":
            reg = alpha * np.insert(np.sign(beta[1:]), 0, 0)
        elif regularization == "l2":
            reg = alpha * np.insert(beta[1:], 0, 0)
        else:
            reg = 0

        beta = beta + learning_rate * (gradient(xs, ys, beta) - reg)
    return beta


class LogisticRegressionModel:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.6,  # Adjust to filter out stop words if necessary
            min_df=5  # Adjust for custom preprocessing
        )

    def preprocess_training_set(self, X, custom=True):
        """Preprocesses the training set. The model is thereby fitted to
        the training data. Uses a custom implementation
        of tf-idf feature extraction if `custom` is True, otherwise uses
        sklearn's TfidfVectorizer.

        `X` is expected to be in the form of a Python list of strings."""

        if custom:
            X = list(map(custom_tokenize, X))

            vocab = set()
            dictnry = dict()
            for tweet in X:
                for word in set(tweet):
                    vocab.add(word)
                    if word not in dictnry:
                        dictnry[word] = 1
                    else:
                        dictnry[word] += 1

            self.vocabulary = list(vocab)
            self.dictionary = dictnry

            n_docs = len(X)
            n_features = len(vocab)
            matrix = np.zeros((n_docs, n_features))
            for i in range(n_docs):
                for j in range(n_features):
                    doc = X[i]
                    word = self.vocabulary[j]
                    tf = doc.count(word) / len(doc)
                    idf = (n_docs + 1) / (self.dictionary[word] + 1)
                    matrix[i, j] = tf * idf

            return matrix
        else:
            return self.vectorizer.fit_transform(X).toarray()

    def preprocess_test_set(self, X, custom=True):
        """Preprocesses the test set using the corpus assembled during
        training. Uses a custom implementation
        of tf-idf feature extraction if `custom` is True, otherwise uses
        sklearn's TfidfVectorizer.

        `X` is expected to be in the form of a Python list of strings."""

        if custom:
            X = list(map(custom_tokenize, X))

            n_docs = len(X)
            n_features = len(self.vocabulary)
            matrix = np.zeros((n_docs, n_features))
            for i in range(n_docs):
                for j in range(n_features):
                    doc = X[i]
                    word = self.vocabulary[j]

                    tf = doc.count(word) / len(doc)
                    idf = (len(self.vocabulary) + 1) / (self.dictionary[word] + 1)
                    matrix[i, j] = tf * idf

            return matrix
        else:
            return self.vectorizer.transform(X).toarray()

    def train(self, xs, ys, n_iters=300, learning_rate=0.01,
              regularization=None, alpha=1):
        """Trains the logistic regression model on the given data,
        executing `n_iters` iterations of gradient descent with
        the given learning rate. The parameter `regularization`
        can be set to either \"l1\" or \"l2\", and `alpha` is the
        rate of regularization.

        `xs` and `ys` are expected to be numpy arrays."""
        xs = np.concatenate((np.ones((xs.shape[0], 1)), xs), axis=1)
        self.beta = reg_gradient_descent(xs, ys,
                                         n_iters,
                                         learning_rate,
                                         regularization=regularization,
                                         alpha=alpha)

    def classify_point(self, x):
        """Classifies the given data point (numpy array).

        The returned value is 0 or 1."""
        xbeta = np.dot(np.concatenate(([1], x)), self.beta)
        return 1 if xbeta >= 0 else 0

    def classify(self, xs):
        """Maps a vector of data points (numpy array) to a vector
        of class labels (0 or 1) by classifying them all."""
        return np.apply_along_axis(self.classify_point, 1, xs)

    def sigmoid(self, x):
        x1 = np.atleast_2d(x)
        x1 = np.concatenate((np.ones((x1.shape[0], 1)), x1), axis=1)
        return np.reciprocal(1 + np.exp(-x1 @ self.beta))


df = pd.read_excel('dataset.xlsx')
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

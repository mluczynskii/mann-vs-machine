import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
import pandas as pd
from src.models.logreg import LogisticRegressionModel


df = pd.read_excel("src/frontend_backend/dataset.xlsx")

X = df.iloc[:, 0].values  # Tweets
y = df.iloc[:, 1].values  # Labels

model = LogisticRegressionModel()

X_preprocessed = model.preprocess_training_set(X, custom=True)

X_preprocessed = np.array(X_preprocessed)

y = np.array(y)

model.train(X_preprocessed, y, n_iters=300, learning_rate=0.01, regularization=None, alpha=1)

model_data = {
    'beta': torch.tensor(model.beta),       # Model parameters (beta)
    'vocabulary': model.vocabulary,         # Vocabulary from training
    'dictionary': model.dictionary          # Dictionary from training
}

torch.save(model_data, 'logRegModel2.pth')

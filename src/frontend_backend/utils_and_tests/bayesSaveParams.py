import numpy as np
import torch
import pickle
from src.models.naive_bayes import NaiveBayesModel
import pandas as pd

df = pd.read_excel("src/frontend_backend/dataset.xlsx")

X = df.iloc[:, 0].values  # Tweets
y = df.iloc[:, 1].values  # Labels

model = NaiveBayesModel()

X_preprocessed = model.preprocess_set(X)

model.train(X_preprocessed, y)

model_data = {
    'dictionary': model.dictionary,   # Word counts for each label
    'n': model.n,                     # Total counts for each label
}

torch.save(model_data, 'naiveBayesModel.pth')

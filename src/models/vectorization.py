import numpy as np

import src.models.tokenization as tkn
from src.models.exception import NotFittedError

class CustomTfidfVectorizer:
    
    
    def __init__(self, min_df=1, max_df_ratio=1):
        pass

    def fit_transform(self, X, splitpascals=False):
        X = list(map(
                lambda x: tkn.tokenize(x, splitpascals=splitpascals),
                X
        ))
        
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

        self.n_docs = len(X)
        n_features = len(vocab)
        matrix = np.zeros((self.n_docs, n_features))
        for i in range(self.n_docs):
            for j in range(n_features):
                doc = X[i]
                word = self.vocabulary[j]

                tf = doc.count(word)/len(doc)
                idf = (self.n_docs + 1)/(self.dictionary[word] + 1)
                matrix[i, j] = tf*idf

        return matrix


    def transform(self, X, splitpascals=False):
        X = list(map(
                lambda x: tkn.tokenize(x, splitpascals=splitpascals),
                X
        ))

        n_test_docs = len(X)

        n_features = len(self.vocabulary)
        if n_features == 0:
            raise NotFittedError(
                """The vectorizer has not been fitted yet.
                You must preprocess a training set first."""
            )
        
        matrix = np.zeros((n_test_docs, n_features))
        for i in range(n_test_docs):
            for j in range(n_features):
                doc = X[i]
                word = self.vocabulary[j]

                tf = doc.count(word)/len(doc)
                idf = (self.n_docs + 1)/(self.dictionary[word] + 1)
                matrix[i, j] = tf*idf

        return matrix

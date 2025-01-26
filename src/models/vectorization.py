import numpy as np

import src.models.tokenization as tkn
from src.models.exception import NotFittedError

class CustomTfidfVectorizer:
    

    n_docs = None

    
    def __init__(self, min_df=1, max_df_ratio=1.0):
        self.min_df = min_df
        self.max_df_ratio = max_df_ratio


    def is_df_within_bounds(self, df):
        if self.n_docs is None:
            raise NotFittedError()
        if self.min_df <= df and df/self.n_docs <= self.max_df_ratio:
            return True
        else:
            return False


    def fit_transform(self, X, splitpascals=False):
        X = list(map(
                lambda x: tkn.tokenize(x, splitpascals=splitpascals),
                X
        ))
        self.n_docs = len(X)

        # build preliminary dictionary
        dictnry = dict()
        for tweet in X:
            for word in set(tweet):
                if word not in dictnry:
                    dictnry[word] = 1
                else:
                    dictnry[word] += 1

        # delete uninformative words from dictionary
        dictnry = {word: df for word, df in dictnry.items() 
                   if self.is_df_within_bounds(df)}

        self.vocabulary = list(dictnry.keys())
        self.dictionary = dictnry

        n_features = len(self.vocabulary)

        # build matrix
        count_matrix = np.zeros((self.n_docs, n_features))
        for i in range(self.n_docs):
            for j in range(n_features):
                doc = X[i]
                word = self.vocabulary[j]

                count_matrix[i, j] = doc.count(word)/(len(doc) + 1)

        v_idf = (self.n_docs + 1)*np.reciprocal(
            np.array([self.dictionary[word] + 1 
                      for word in self.vocabulary]
            )
        )

        matrix = count_matrix*v_idf

        # normalize vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix/np.where(norms == 0, 1, norms)

        return matrix


    def transform(self, X, splitpascals=False):
        X = list(map(
                lambda x: tkn.tokenize(x, splitpascals=splitpascals),
                X
        ))

        n_test_docs = len(X)

        n_features = len(self.vocabulary)
        if n_features == 0:
            raise NotFittedError()
        
        # build matrix
        count_matrix = np.zeros((n_test_docs, n_features))
        for i in range(n_test_docs):
            for j in range(n_features):
                doc = X[i]
                word = self.vocabulary[j]

                count_matrix[i, j] = doc.count(word)/len(doc)

        v_idf = (self.n_docs + 1)*np.reciprocal(
            np.array([self.dictionary[word] + 1 
                      for word in self.vocabulary]
            )
        )

        matrix = count_matrix*v_idf

        # normalize vectors
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        matrix = matrix/np.where(norms == 0, 1, norms)

        return matrix

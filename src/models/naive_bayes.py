import numpy as np

import src.models.tokenization as tkn
from src.models.exception import NotTrainedError


class NaiveBayesModel:


    def preprocess_tweet(_, x, splitpascals=False):
        """Tokenizes a string."""

        return tkn.tokenize(x, returntype="set", splitpascals=splitpascals)
    

    def preprocess_set(self, X):
        """Tokenizes all data points in the given dataset.
        
        `X` is expected to be a regular Python list."""

        return list(map(self.preprocess_tweet, X))


    def train(self, X, y):
        """Trains the model on the given data. 

        `X` and `y` are expected to be regular Python lists."""
       
        d = dict()
        n = [0, 0]

        for tweet, label in zip(X, y):
            for word in tweet:
                if word in d:
                    d[word][label] += 1
                    n[label] += 1
                else:
                    d[word] = [0, 0] 
                    d[word][label] += 1

        self.dictionary = d
        self.n = n
        self.custom_trained = True


    def classify_tweet(self, x):
        """Classifies a data point given as a set of words
        (as tokenized by preprocess_tweet)."""

        if not self.custom_trained:
            raise NotTrainedError(
                "This Naive Bayes model has not been trained yet"
            )

        coeff = [0, 0]
        for label in [0, 1]:
            coeff[label] = (1 - len(x))*np.log(self.n[label])
            - np.log(self.n[0] + self.n[1])

            for word in x:
                if word in self.dictionary:
                    count = self.dictionary[word][label]
                    if count != 0:
                        coeff[label] += np.log(self.dictionary[word][label])

        return 1 if coeff[1] >= coeff[0] else 0

    
    def classify(self, X):
        """Classifies all data points in the data set given
        as Python list of sets of words."""

        return list(map(self.classify_tweet, X))
    
import numpy as np
import src.models.tokenization as tkn
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer


class RandomForestModel:

    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.trees = []
        self.vectorizer = TfidfVectorizer(sublinear_tf=True, max_df=0.6, min_df=5)

    def preprocess_tweet(self, x, splitpascals=False):
        """Tokenizes a string."""
        return tkn.tokenize(x, returntype="set", splitpascals=splitpascals)

    def preprocess_set(self, X):
        """Tokenizes all data points in the given dataset."""
        return list(map(self.preprocess_tweet, X))

    def train(self, X, y):
        """Trains the Random Forest model using TF-IDF."""
        # Converting to a string might seem pointless at first glance, but vectorizer expects a collection of strings as argument.
        X = [" ".join(tweet) for tweet in X]

        # Convert the tokenized text to numerical features
        X_tfidf = self.vectorizer.fit_transform(X).toarray()

        for _ in range(self.n_estimators):
            X_resampled, y_resampled = resample(X_tfidf, y, random_state=42)

            tree = DecisionTreeClassifier(max_depth=self.max_depth)
            tree.fit(X_resampled, y_resampled)
            self.trees.append(tree)

    def classify_tweet(self, x):
        """Classifies a data point using the trained Random Forest model."""
        x_tfidf = self.vectorizer.transform([" ".join(x)]).toarray()
        predictions = [tree.predict(x_tfidf)[0] for tree in self.trees]

        return 1 if sum(predictions) >= len(predictions) / 2 else 0

    def classify(self, X):
        """Classifies all data points in the dataset."""
        return list(map(self.classify_tweet, X))
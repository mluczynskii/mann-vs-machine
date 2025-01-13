import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

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
            reg = alpha*np.insert(np.sign(beta[1:]), 0, 0)
        elif regularization == "l2":
            reg = alpha*np.insert(beta[1:], 0, 0)
        else:
            reg = 0

        beta = beta + learning_rate*(gradient(xs, ys, beta) - reg) 
    return beta

        
class LogisticRegressionModel:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            sublinear_tf=True,
            max_df=0.6, # should it be more, so that stop words are filtered out?
            min_df=5
        )


    # We use sklearn.TfidfVectorizer while preprocessing.
    # We could replace it by our own implementation of
    # tf-idf vectorization. It shouldn't be too difficult.
    def preprocess_training_set(self, X):
        return self.vectorizer.fit_transform(X)
    
    
    def preprocess_test_set(self, X):
        return self.vectorizer.transform(X)
        

    def train(self, xs, ys, n_iters=300, learning_rate=0.001,
               regularization=None, alpha=1):
        xs = np.concatenate((np.ones((xs.shape[0], 1)), xs), axis=1)
        self.beta = reg_gradient_descent(xs, ys,
                                         n_iters,
                                         learning_rate,
                                         regularization=regularization, 
                                         alpha=alpha)


    def classify_point(self, x):
        xbeta = np.dot(np.concatenate(([1], x)), self.beta)
        return 1 if xbeta >= 0 else 0
    

    def classify(self, xs):
        return np.apply_along_axis(self.classify_point, 1, xs)
    

    def sigmoid(self, x): 
        x1 = np.atleast_2d(x)
        x1 = np.concatenate((np.ones((x1.shape[0], 1)), x1), axis=1)
        return np.reciprocal(1 + np.exp(-x1 @ self.beta))
    
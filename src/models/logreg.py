import numpy as np

from src.models.exception import NotTrainedError

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


    custom_trained = False
        

    def train(self, xs, ys, n_iters=300, learning_rate=0.001,
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
        self.custom_trained = True


    def classify_point(self, x):
        """Classifies the given data point (numpy array).
        
        The returned value is 0 or 1."""

        if not self.custom_trained:
            raise NotTrainedError(
                "This custom Logistic Regression model has not been trained yet."
            )

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
    
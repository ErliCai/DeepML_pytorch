import numpy as np

def train(X, y, W, b):
    """
    Train linear regression weights on standardized data.
    
    Args:
        X: numpy array of shape (n_samples, n_features) -- standardized features
        y: numpy array of shape (n_samples,) -- standardized targets
        W: numpy array of shape (n_features,) -- initial random weights
        b: float -- initial bias (0.0)
    
    Returns:
        W: numpy array of shape (n_features,) -- trained weights
        b: float -- trained bias
    """
    # TODO: implement your training strategy here
    # You can use ANY approach: gradient descent, normal equation,
    # momentum, adaptive learning rates, mini-batching, etc.

    # learning_rate = 0.001
    # for j in range(3):
    #     for i in range(len(X)):
    #         prediction = np.dot(X[i], W) + b
    #         loss = prediction - y[i]

    #         W = W - learning_rate * 2 * X[i] * loss
    #         b = b - learning_rate * 2 * loss
    # return W, b

    learning_rate = 0.001
    epoch = 1000
    n = len(X)
    for i in range(epoch):
        prediction = X @ W + b
        error = (prediction - y)

        W = W - learning_rate * (2 / n) * (X.T @ error)
        b = b - learning_rate * (2 / n) * np.sum(error)

    return W, b


import numpy as np


def self_attention(X: np.ndarray) -> np.ndarray:
    """
    Compute simple self-attention without trainable weights.

    Args:
        X: numpy array of shape (seq_len, d_model) representing the input sequence,
           used directly as queries, keys, and values.

    Returns:
        numpy array of shape (seq_len, d_model) with the attention output.
    """

    X = np.array(X)

    scores = X @ X.T
    softmax = [np.exp(row)/sum(np.exp(row)) for row in scores]
    context_vector =  np.zeros_like(X, dtype=float)
    for i in range(len(X)):
        context_vector[i] = softmax[i] @ X

    return context_vector

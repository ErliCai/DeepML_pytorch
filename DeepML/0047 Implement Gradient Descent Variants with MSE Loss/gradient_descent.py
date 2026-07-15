import torch

def gradient_descent(X: torch.Tensor, y: torch.Tensor, weights: torch.Tensor, 
                    learning_rate: float, n_epochs: int, 
                    batch_size: int = 1, method: str = 'batch') -> torch.Tensor:
    """
    Implements three variants of gradient descent: Batch, Stochastic, and Mini-Batch.
    Uses Mean Squared Error (MSE) as the loss function.
    
    Args:
        X: Feature matrix of shape (m, n)
        y: Target values of shape (m,)
        weights: Initial weights of shape (n,)
        learning_rate: Step size for gradient descent
        n_epochs: Number of complete passes through the dataset
        batch_size: Size of batches for mini-batch gradient descent (default: 1)
        method: Type of gradient descent ('batch', 'stochastic', or 'mini_batch')
    
    Returns:
        Optimized weights as a tensor
    """

    m = len(y)

    for _ in range(n_epochs):
        if method == "batch":
                y_pred = X @ weights
                weights = weights - learning_rate * 2 / m * X.T @ (y_pred - y)
        
        if method == "stochastic":
                for i in range(m):
                    y_i_pred = X[i,:] @ weights
                    weights = weights - learning_rate * 2 * (y_i_pred - y[i]) * X.T[:,i]

        if method == 'mini_batch':
            for i in range((m-1)//batch_size+1):
                sample_start = i * batch_size
                sample_end =  min((i+1) * batch_size, m)
                size = sample_end - sample_start

                y_pred = X @ weights

                weights  = weights - learning_rate * 2 / size * X.T[:, sample_start:sample_end] @ (y_pred - y)[sample_start:sample_end]

    return weights
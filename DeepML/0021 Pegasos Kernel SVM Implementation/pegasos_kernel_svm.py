import torch

def pegasos_kernel_svm(data: torch.Tensor, labels: torch.Tensor, kernel: str = 'linear', lambda_val: float = 0.01, iterations: int = 100, sigma: float = 1.0) -> tuple:
    """
    Train a kernel SVM using the deterministic Pegasos algorithm.
    
    Args:
        data: Training data of shape (n_samples, n_features)
        labels: Labels of shape (n_samples,) with values in {-1, 1}
        kernel: 'linear' or 'rbf'
        lambda_val: Regularization parameter
        iterations: Number of training iterations
        sigma: RBF kernel bandwidth (only used if kernel='rbf')
    
    Returns:
        Tuple of (alphas, bias) where alphas is a list and bias is a float
    """

    # initialize coefficient
    alpha = [0] * len(labels)
    beta = 0

    if kernel == 'linear':
        kernal_func = linear_kernel
    elif kernel == 'rbf':
        kernal_func = rbf_kernel

    for iteration in range(1, iterations+1):
        learning_rate = 1/(lambda_val * iteration)
        decision_value = [0] * len(data)
        for i in range(len(labels)):
            decision_value[i] = sum([alpha[j] * labels[j] * kernal_func(data[i], data[j], sigma)  for j in range(len(data))]) + beta

            if labels[i] * decision_value[i] < 1:
                alpha[i] = (1 - learning_rate * lambda_val) * alpha[i] + learning_rate * labels[i]
                beta = beta + learning_rate * labels[i]
            # else:
            #     alpha[i] *= (1 - learning_rate * lambda_val)

    print(alpha, beta)
    return [float(a) for a in alpha], beta.item()


def linear_kernel(x, y, sigma):

    return torch.dot(x, y)

def rbf_kernel(x, y ,sigma):

    return torch.exp( - torch.sum((x - y) ** 2) / (2 * sigma ** 2))
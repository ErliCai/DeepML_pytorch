import torch

def kl_divergence_normal(mu_p, sigma_p, mu_q, sigma_q) -> torch.Tensor:
    """
    Compute the KL divergence between two normal distributions P and Q.
    
    Args:
        mu_p: Mean of distribution P
        sigma_p: Standard deviation of distribution P
        mu_q: Mean of distribution Q
        sigma_q: Standard deviation of distribution Q
    
    Returns:
        torch.Tensor: KL divergence KL(P || Q)
    """
    sigma_p = torch.tensor(sigma_p)
    sigma_q = torch.tensor(sigma_q)

    rst = torch.log(sigma_q/sigma_p) + (sigma_p ** 2 + (mu_p-mu_q) ** 2)/(2 * sigma_q ** 2) - 1/2
    return torch.tensor(rst)
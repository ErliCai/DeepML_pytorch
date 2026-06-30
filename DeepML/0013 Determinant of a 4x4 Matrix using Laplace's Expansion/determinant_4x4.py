import torch

def determinant_4x4(matrix) -> float:
    """
    Compute the determinant of a 4×4 matrix using PyTorch.
    Input can be a Python list, NumPy array, or torch Tensor of shape (4,4).
    Returns a Python float.
    """
    # Convert to tensor
    m = torch.as_tensor(matrix, dtype=torch.float)
    
    det = 0
    for i in range(4):
        a = m[i,0]
        m_3 = m[:,1:]
        m_3 = torch.cat([m_3[:i,:], m_3[i+1:,:]],dim=0)
        det += ((-1) ** i)  * a * torch.det(m_3)

    return det.item()
"""
Kernels for Gaussian Process Regression
"""
import torch
torch.set_default_tensor_type(torch.DoubleTensor)

def covariance(X,phi,tau2):
    """
    Description
    -----------
    Calculates the covariance matrix.
    tau2 exp(-0.5*phi^2 (x1-x2)^2)
    
    Parameters
    ----------
    X : (n x m) torch tensor or a tuple of (n1 x m) and (n2 x m) torch tensors
        The n rows are the different datapoints and the m columns represent
        the different features
    phi : scalar or (m,) shaped torch tensor of inverse length-scales 
    tau2 : Amplitude of the squared exponential kernel
    
    Returns
    -------
    cov : (n x n) torch tensor
    cov_grad : (n x n x m) torch tensor with gradient
    
    """
    phi = phi[None,:]
    if type(X)!=tuple:
        x=X
        y=X
    else:
        x=X[0]
        y=X[1]
    
    result = tau2*torch.exp(-0.5*((torch.cdist(x*phi,y*phi))**2))
    n = result.shape[0]
    result.view(1,-1)[:,::n+1] += 1.e-5
    return result


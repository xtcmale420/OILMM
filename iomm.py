import pdb
import numpy as np
import scipy
import torch
import torch.nn as nn
from torch.distributions.multivariate_normal import MultivariateNormal
from base import enforce_type 

class OrthogonalMixingModel(nn.Module):

    def __init__(self, obs_dim, latent_dim, base_models, U=None, logS=None, sigma_sq=None, logD=None):
        super().__init__()
        
        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        if U is None:
            U = np.random.rand(self.obs_dim, self.latent_dim)
            U, _ = np.linalg.qr(U)
            U = enforce_type(U, torch.Tensor)
        else:
            U = enforce_type(U, torch.Tensor)
        if logS is None:
            logS = torch.ones(self.latent_dim, self.latent_dim)
        else:
            logS = enforce_type(logS, torch.Tensor)
        if sigma_sq is None:
            sigma_sq = torch.tensor(1.0)
        else:
            sigma_sq = enforce_type(sigma_sq, torch.Tensor)
        if logD is None:
            logD = torch.ones(self.latent_dim, self.latent_dim)
        else:
            logD = enforce_type(logD, torch.Tensor)
        self.base_models = nn.ModuleList(base_models)
        self.U = nn.Parameter(U)
        self.logS = nn.Parameter(logS)
        self.sigma_sq = nn.Parameter(sigma_sq)
        self.logD = nn.Parameter(logD)

    # generate samples from the currently fit model
    def sample(self, n_s):
        # Requires base_models to have a sample function implemented
        x = torch.cat([torch.unsqueeze(model.sample(n_s), 1) for model in self.base_models], 1)
        # Sample observational noise in the projected space
        
        sigma_d = MultivariateNormal(torch.zeros(len(self.base_models)), torch.diag(torch.exp(self.logD))).sample(torch.Size([n_s]))    
        # H = U S^1/2
        H = self.U @ torch.diag(torch.exp(1/2 * self.logS))
        y = torch.matmul(x, torch.t(H))
        
        # Observational noise in the output space
        sigma_ = MultivariateNormal(torch.zeros(y.shape[-1]), self.sigma_sq * torch.eye(y.shape[-1])).sample(torch.Size([n_s]))
        y += sigma_
        return y
    
    # Condition the likelihood on the orthogonal combinations of the observed data to speed up multi-output inference
    def OrthogonalMixingLikelihood(self, Y):

        # Form the summary statistics
        H = torch.matmul(self.U, torch.diag(torch.pow(torch.exp(self.logS), -1)))
        yproj = torch.matmul(Y, H)

        # Observation noise
        Sigma_T = self.sigma_sq * torch.pow(torch.exp(self.logS), -1) + torch.exp(self.logD) 

        # Log marginal likelihoods of the individual base problems
        lml = torch.sum(*[base_model.marginal_likelihood(yproj, Sigma_T) for base_model in self.base_models])
    
        # Keeping notation from OLMM paper, m is the number of latent processes
        n = Y.shape[0]
        p = self.U.shape[0]
        m = self.U.shape[1]
        reg = -n/2 * torch.abs(torch.prod(self.logS)) - n * (p - m)/2 * torch.log(2 * np.pi * self.sigma_sq) - \
              1/(2 * self.sigma_sq) * (torch.pow(torch.linalg.norm(Y), 2) - torch.pow(torch.linalg.norm(torch.matmul(Y, self.U)), 2))
        return reg + lml
    
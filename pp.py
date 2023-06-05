import numpy as np
import torch
from torch import nn
from base import enforce_type, CosineKernel, BaseGP

# Chebyshev approximation level 1
def get_chebyshev_coef(f, x0, x1, m):
    
    nx = 1000
    x = np.linspace(x0, x1, nx)
    y = f(x)
    c = np.polynomial.chebyshev.Chebyshev.fit(x, y, deg=m, domain=(x0, x1))
    return np.polynomial.chebyshev.cheb2poly(c.coef)

# Chebyshev approximation level 2
class CountDistribution():

    def __init__(self):
        pass
    
    @classmethod
    def f(self, x):
        raise NotImplementedError

    @classmethod
    def approximate(cls, x0, x1):
        x0 = np.array(x0)
        x1 = np.array(x1)
        
        a = torch.zeros(x0.shape[0])
        b = torch.zeros(x0.shape[0])
        c = torch.zeros(x0.shape[0])
        for j in range(x0.shape[0]):
            coef_ = get_chebyshev_coef(cls.f, x0[j], x1[j], 2)
            c[j] = torch.tensor(coef_[0])
            b[j] = torch.tensor(coef_[1])
            a[j] = torch.tensor(coef_[2])

        return a, b, c

# Chebyshev approximation level 3    
class PoissonCountDistribution(CountDistribution):

    def __init__(self):
        pass

    @classmethod
    def f(cls, x):
        return np.exp(x)

    @classmethod
    def approximate(cls, y):
        y = np.array(y)
        
        x0 = []
        for i in range(y.shape[1]):
            x0_i = np.mean(y[:, i]) - 2
            x0.append(x0_i)
        x1 = []
        for i in range(y.shape[1]):
            x1_i = np.mean(y[:, i]) + 2
            x1.append(x1_i)
        return super().approximate(x0, x1)

# Uses Cheybshev approximations to compute the marginal likelihood
class PointProcessGPFA(nn.Module):

    def __init__(self, obs_dim, latent_dim, base_models, W=None):

        super(PointProcessGPFA, self).__init__()

        self.obs_dim = obs_dim
        self.latent_dim = latent_dim
        self.count_dist = PoissonCountDistribution()
        
        if W is None:
            W = torch.randn(self.obs_dim, self.latent_dim)
        else:
            assert(W.shape[0] == self.obs_dim)
            W = enforce_type(W, torch.Tensor)
        
        self.W = nn.Parameter(W)
        self.base_models = nn.ModuleList(base_models)
    
    def sample(self, T):
        # Requires base_models to have a sample function implemented
        x = torch.cat([torch.unsqueeze(model.sample(T), 1) for model in self.base_models], 1)
        lambd_ = torch.exp(x @ self.W.T)
        y = torch.cat([torch.unsqueeze(torch.poisson(lambd_[:, i]), 1) for i in range(self.obs_dim)], 1)
        return y.detach().numpy()
 
    # Cheybshev approximations
    def get_chebyshev_coefficients(self, y):
        a, b, c = self.count_dist.approximate(y)

        self.a = a
        self.b = b
        self.c = c

    def approximate_marginal_likelihood(self, y):
        if not hasattr(self, 'a'):
            self.get_chebyshev_coefficients(y)
        y = np.array(y)        
        assert(y.shape[1] == self.obs_dim)
        T = y.shape[0]
        K = torch.block_diag(*[m.get_cov_matrix(T) for m in self.base_models])
        I = torch.eye(T)
        W1 = torch.kron(self.W, I)

        a = torch.kron(self.a, torch.ones(T))
        b = torch.kron(self.b, torch.ones(T))
        
        y = torch.tensor(y).transpose(0, 1).ravel()

        # Need to reshape appropriately
        A = torch.diag(a)
        S = torch.chain_matmul(torch.t(W1), A, W1)
        
        Sigma_inv = torch.multiply(2, S) + torch.linalg.inv(K)
        mu = torch.matmul(torch.linalg.inv(Sigma_inv), torch.matmul(torch.t(W1), torch.subtract(y, b)))
        term1 = 0.5 * torch.slogdet(Sigma_inv)[1]
        term2 = 0.5*torch.chain_matmul(torch.unsqueeze(mu, 0), Sigma_inv, torch.unsqueeze(mu, 1))[0, 0]
        
        term3 = -1*0.5*torch.slogdet(K)[1]

        log_likelihood = term1 + term2 - term3
        
        return log_likelihood

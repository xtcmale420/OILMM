import numpy as np
import torch
from iomm import OrthogonalMixingModel
from base import BaseGP, CosineKernel
from train import train_loop, Cross_Validation_iomm

# Suppose that the latent dimension is 3, here are the parameters
U = np.array([[1,0,0], [0,1,0], [0,0,1], [0,0,0]])
logS = np.array([1, 1, 1])
logD = np.array([1, 1, 1])
sigma_sq = np.array(1)
sigma = [3.58804129, 2.29182921, 3.44943223]
mu = [2.50000134, 1.98118901e-2, 2.49998292]
T = 100

# Generate a list of data
base_model = [BaseGP(CosineKernel, sigma=sigma[i], mu=mu[i]) for i in range(3)]
PG0 = OrthogonalMixingModel(4, 3, base_model, U, logS, sigma_sq, logD)
y = PG0.sample(T)

# Find out the best latent dimension
Cross_Validation_iomm(y)
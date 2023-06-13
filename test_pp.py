import numpy as np
import torch
from pp import PointProcessGPFA
from base import BaseGP, CosineKernel
from train import train_loop, Cross_Validation_pp

# Suppose that the latent dimension is 3, here are the parameters
W = np.array([[0.2481, 0.3873, 0.1202], [0.2471, 0.3856, 0.1197], [0.2477, 0.3867, 0.1200], [0.2481, 0.3873, 0.1202]])
sigma = [3.58804129, 2.29182921, 3.44943223]
mu = [2.50000134, 1.98118901e-2, 2.49998292]
T = 100

# Generate a list of data
base_models = [BaseGP(CosineKernel, sigma=sigma[i], mu=mu[i]) for i in range(3)]
PG0 = PointProcessGPFA(4, 3, base_models, W)
y = PG0.sample(T)

Cross_Validation_pp(y)

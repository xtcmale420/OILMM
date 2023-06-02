import torch
from sklearn.model_selection import KFold
from base import BaseGP, CosineKernel, enforce_type
from pp import PointProcessGPFA
from iomm import OrthogonalMixingModel

# Example structure of a training loop within pytorch
def train_loop(y, model, loss):
    
    # Training process
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    for i in range(10):
        optimizer.zero_grad()
        l = -1*loss(y)
        print('l:%f' % l)
        # This takes the calculate value, and automatically calculates gradients with respect to parameters
        l.backward(retain_graph=True)
        # Optimizer will take the gradients, and then update parameters accordingly
        optimizer.step()
        # Calculate new loss given the parameter update
        l1 = -1*loss(y).detach()
        delta_loss = torch.abs(l1 - l)
        print('l1:%f' % l1)
    
    parameters = [param for param in model.parameters()]
    return parameters
             
# Cross-Validation of pp model
def Cross_Validation_pp(y):
    
    # Divid the data
    sk = KFold(10)
    sk = sk.split(y)
    marginal_log_likelihood = torch.tensor(0)
    dimension = 0
    
    # Check each possible latent dimension
    for i in range(1, y.shape[1]):
        
        log_likelihood = torch.tensor(0.0)
        
        for train, test in sk:
            
            traindata = y[train]
            testdata = y[test]
            
            base_models = [BaseGP(CosineKernel) for j in range(i)]
            PG0 = PointProcessGPFA(4, i, base_models)
            
            parameters = train_loop(traindata, PG0, PG0.approximate_marginal_likelihood)
            W = parameters[0]
            sigma = []
            mu = []
            for t, _ in enumerate(parameters):
                if t==0:
                    pass
                elif t%2==1:
                    sigma.append(_)
                elif t%2==0:
                    mu.append(_)
            
            base_models = [BaseGP(CosineKernel, sigma=sigma[j], mu=mu[j]) for j in range(i)]
            PG0 = PointProcessGPFA(4, 3, base_models, W) 
            likelihood = PG0.approximate_marginal_likelihood(testdata)
            log_likelihood += likelihood
            
        log_likelihood = torch.div(log_likelihood, 10) 
        if log_likelihood <= marginal_log_likelihood:
            marginal_log_likelihood = log_likelihood
            dimension = i
            
        print('The most possible latent dimension is %f' % dimension)
        
def Cross_Validation_iomm(y):
        
    # Divid the data
    sk = KFold(10)
    sk = sk.split(y)
    marginal_log_likelihood = torch.tensor(0)
    dimension = 0
        
    # Check each possible latent dimension
    for i in range(1, y.shape[1]):
        
        log_likelihood = torch.tensor(0.0)
        
        for train, test in sk:
            
            traindata = y[train]
            testdata = y[test]
            
            base_models = [BaseGP(CosineKernel) for j in range(i)]
            PG0 = OrthogonalMixingModel(4, i, base_models)
            
            parameters = train_loop(traindata, PG0, PG0.OrthogonalMixingLikelihood)
            U = parameters[0]
            logS = parameters[1]
            sigma_sq = parameters[2]
            logD = parameters[3]
            sigma = []
            mu = []
            for t in range(4, len(parameters)):
                if t%2==0:
                    sigma.append(parameters[t])
                else:
                    mu.append(parameters[t])

            base_models = [BaseGP(CosineKernel, sigma=sigma[j], mu=mu[j]) for j in range(i)]
            PG0 = OrthogonalMixingModel(4, 3, base_models, U, logS, sigma_sq, logD) 
            likelihood = PG0.OrthogonalMixingLikelihood(testdata)
            log_likelihood += likelihood
            
        log_likelihood = torch.div(log_likelihood, 10) 
        if log_likelihood <= marginal_log_likelihood:
            marginal_log_likelihood = log_likelihood
            dimension = i
            
        print('The most possible latent dimension is %f' % dimension)
        
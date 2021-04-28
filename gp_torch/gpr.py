import torch
import math
from .kernels import *
torch.set_default_tensor_type(torch.DoubleTensor)
torch.pi = torch.tensor(torch.acos(torch.zeros(1)).item() * 2)

class gaussian_process_regressor:
    """
    Gaussian Process regression using torch autograd for parameter tuning
    """
    def __init__(self,x_train=torch.ones((0,1)),y_train=torch.zeros((0,1)),kernel='sq_exp',optimizer='Adam',prior='ard',normalize_x=False,normalize_y=False,n_restarts_optimizer = 0):
        """
        Creates the model.
        Currently the rest of this file assumes a squared-exponential kernel, Adam optimizer and n_restarts_optimizer = 0.
        The arguments are just placeholders for future extensions
        """
        super().__init__()
        self.x_train = x_train
        self.y_train = y_train
        self.kernel = kernel
        self.n_train = x_train.shape[0]
        self.x_dim = x_train.shape[1]
        self.phi = torch.ones(x_train.shape[1], requires_grad=True)
        self.tau = torch.tensor(1.,requires_grad=True)
        #=============================================================================================================================================
        # Something wrong with noise. Check gpedm/interactive/example.py. Fixing noise to a static (unoptimizable) value leads to reasonable outputs. 
        # But leaving noise as trainable leads to strange outputs
        #=============================================================================================================================================
        self.noise = torch.tensor(1.e-5,requires_grad=True)
        self.optimizer = torch.optim.Adam([self.phi,self.tau,self.noise], lr=0.001)
        self.prior = prior
        self.n_restarts_optimizer = n_restarts_optimizer
        #===============================================================================================================================================
        # Something wrong with normalization. Leads to horrible predictions when normalization turned on. Check gpedm/interactive/example.py
        # Consider removing normalization as this breaks the flow of the rest of the code. Normalization can be done externally. See gpr_known_noise
        # CONSIDER MERGING THIS WITH GPR_KNOWN_NOISE.PY WITH AN OPTION TO KEEP NOISE STATIC. DO THIS ONLY AFTER FIXING NORMALIZATION PROBLEM
        #===============================================================================================================================================
        self.normalize_x=normalize_x
        self.normalize_y=normalize_y
        #if self.prior=='ard' and (self.normalize_x==False or self.normalize_y==False):
        #    self.normalize_x=True
        #    self.normalize_y=True
        #    print("Setting internal normalization of x_train and y_train to True to be consistent with ARD prior specs")
        
        # Normalize x and y (zero mean, unit variance per column)
        if self.normalize_x:
            self.x_train_std, self.x_train_mean = torch.std_mean(x_train, axis=0)
            if torch.isnan(self.x_train_std):
                self.x_train_std = 1.
            self.x_train = (self.x_train - self.x_train_mean)/self.x_train_std
        else:
            self.x_train_mean = torch.zeros(x_train.shape[1])
            self.x_train_std = torch.ones(x_train.shape[1])
        if self.normalize_y:
            self.y_train_std, self.y_train_mean = torch.std_mean(y_train, axis=0)
            if torch.isnan(self.y_train_std):
                self.y_train_std = 1.
            self.y_train = (self.y_train - self.y_train_mean)/self.y_train_std            
        else:
            self.y_train_mean = torch.zeros(y_train.shape[1])
            self.y_train_std = torch.ones(y_train.shape[1])
        
        if self.prior=='ard' and self.kernel=='sq_exp': #generalize this later on
            #self.prior_phi = torch.distributions.half_normal.HalfNormal(torch.tensor([torch.pi/torch.sqrt(torch.tensor(12.))])) # Has zero probability for phi < 0., this is problematic when phi close to zero
            if self.normalize_x:
                phi_std = torch.sqrt(torch.pi/torch.sqrt(torch.tensor(12.)))
            else:
                phi_std = self.x_train_std*torch.sqrt(torch.pi/torch.sqrt(torch.tensor(12.)))
            self.prior_phi = torch.distributions.normal.Normal(0.,torch.tensor([phi_std]))
            self.prior_tau2 = torch.distributions.gamma.Gamma(torch.tensor([1.]),torch.tensor([1.])) #Note torch Gamma parameterized by alpha (shape) and beta (rate, not scale)
            self.prior_noise2 = torch.distributions.gamma.Gamma(torch.tensor([1.]),torch.tensor([1.])) #beta(a=1.1,b=1.1) causes log_prior = -inf at initial_noise2=1.
        
        
    
    def log_marginal_likelihood(self):
        """
        Calculates the log marginal likelihood of y_train given x_train
        """
        K = covariance(self.x_train, self.phi, self.tau**2)
        K_noisy = K.clone()
        K_noisy.view(-1)[::self.n_train+1] += (self.noise**2)
        L = K_noisy.cholesky()
        alpha = torch.cholesky_solve(self.y_train,L)
        log_likelihood_dims = -0.5*(self.y_train*alpha).sum(axis=0) - torch.log(L.trace()) - 0.5*self.n_train*torch.log(2.*torch.pi)
        return log_likelihood_dims.sum()
    
    def objective_function(self):
        """
        Calculates the objective function to be minimized during optimization of parameters. Currently supports no priors or ARD prior
        """
        if self.prior == None:
            return -self.log_marginal_likelihood()
        elif self.prior == 'ard':
            log_posterior = self.log_marginal_likelihood() + self.prior_noise2.log_prob(self.noise**2) + self.prior_tau2.log_prob(self.tau**2) + self.prior_phi.log_prob(self.phi).sum()
            return -log_posterior
    
    def predict(self,x_new,return_std=True):
        """
        Predict y_mean(x_new | x_train, y_train). Can be called before or after optimization of parameters
        """
        if self.normalize_x:
            x_new = (x_new-self.x_train_mean)/self.x_train_std
        K = covariance(self.x_train, self.phi, self.tau**2)
        K_noisy = K.clone()
        K_noisy.view(-1)[::self.n_train+1] += (self.noise**2)
        K_trans = covariance((x_new,self.x_train), self.phi, self.tau**2)
        L = K_noisy.cholesky()
        alpha = torch.cholesky_solve(self.y_train,L)
        y_mean = K_trans @ alpha
        if return_std:
            v = torch.cholesky_solve(K_trans.T,L)
            y_cov = covariance(x_new, self.phi, self.tau**2) - K_trans @ v
            y_std = torch.sqrt(torch.diag(y_cov)).reshape(-1,1) 
        if self.normalize_y:
            y_mean = self.y_train_std*y_mean + self.y_train_mean
            if return_std:
                y_std = self.y_train_std*y_std
        if return_std:
            return y_mean, y_std
        else:
            return y_mean
    
    def optimize(self,iterations=10000):
        """
        Optimize parameters to minimize self.objective_function()
        """
        for t in range(iterations):
            objective = self.objective_function()
            if t%1000 == 0:
                print(t,objective.item())
            self.optimizer.zero_grad()
            objective.backward()
            self.optimizer.step()


# -*- coding: utf-8 -*-

import numpy as np
import scipy.stats as sts 
from scipy.stats import norm

## Class SA taken from https://github.com/AchrafTamtalini/SRM/tree/main/Simulation%20MSRM/MC
class SA(object):
    def __init__(self, X, c, gamma, K, t, init, epsilon):
        #parameter t for the PR
        self._t = t
        #compact set
        self._K = K  
        #realizations of the random vector X
        self._X = X     
        self._dim = X.shape[1]
        self._maxIter = X.shape[0]
        #parameter c in step sequence gamma_n = c / n ** gamma
        self._c = c
        #paramater gamma in step sequence gamma_n = c / n ** gamma
        self._gamma = gamma
        #initial value for our SA algorithm
        self._init = init
        #for the estimation of covariance and jacobian matrix
        self._epsilon = epsilon
        self._sigmaEst, self._jacEst = np.zeros((self._dim, self._dim)), np.zeros((self._dim, self._dim))
        #sequence of the algorithm
        self._z = np.zeros((self._maxIter, self._dim + 1))
   
        
    @property
    def X(self):
        return self._X
    
    @property
    def c(self):
        return self._c
    
    @property
    def gamma(self):
        return self._gamma
    
    @property
    def K(self):
        return self._K
    
    @property 
    def t(self):
        return self._t
    
    @property
    def init(self):
        return self._init
    
    @property
    def epsilon(self):
        return self._epsilon
    
    @property
    def maxIter(self):
        return self._maxIter
    
    @property
    def dim(self):
        return self._dim
    
    @property
    def z(self):
        return self._z
    
    
    @property
    def jacEst(self):
        return self._jacEst
    
    @property
    def sigmaEst(self):
        return self._sigmaEst
    
    @sigmaEst.setter
    def sigmaEst(self, sigmaEst):
        self._sigmaEst = sigmaEst
        
    @jacEst.setter
    def jacEst(self, jacEst):
        self._jacEst = jacEst
        
    
    
    def check_gamma(self):
        if self._gamma > 1 or self._gamma <= 0.5:
            raise ValueError('The value of gamma is not accepted')
        return self._gamma
    
    def projection(self, m):
        for i in range(self._dim + 1):
            if m[i] < self._K[i][0]:
                m[i] = self._K[i][0]
            if m[i] > self._K[i][1]:
                m[i] = self._K[i][1]
        return m
    
    def setRM(self):
        gamma = self.check_gamma()
        z = np.zeros((self._dim + 1, self._maxIter))
        z[:, 0] = self._init
        for i in range(1, self._maxIter):
            z[:,i] = self.projection(z[:,i - 1] + (self._c / (i ** gamma)) * self.H(self.X[i], z[:,i - 1]))
        self._z = z
        return z
            
    def setEst(self):
        gamma = self.check_gamma()
        z = self._z
        sigmaEst = np.zeros((self._dim + 1, self._dim + 1))
        jacEst = np.zeros((self._dim + 1, self._dim + 1))
        I = np.identity(self._dim + 1)
        for i in range(1, self._maxIter):
            sigmaEst += np.outer(self.H(self.X[i], z[:,i - 1]), self.H(self.X[i], z[:,i - 1]))
            for j in range(self._dim + 1):
                jacEst[:,j] += (self.H(self.X[i], z[:,i - 1] + self._epsilon * I[:,j]) - self.H(self.X[i], z[:,i - 1])) / self._epsilon
        self.sigmaEst = sigmaEst / self._maxIter 
        self.jacEst = jacEst / self._maxIter

    
    #Before calling getPR, we should call setEst first
    def getPR(self):
        if self._gamma == 1:
            raise ValueError('Value of gamma is not less than 1')
        else:
            gamma = self.check_gamma()
            initIndex = int(self._maxIter - (self._t / self._c) * (self._maxIter ** gamma))
            if initIndex < 0:
                initIndex = 0
            invA = np.linalg.inv(self.jacEst)
            V = np.dot(invA, np.dot(self.sigmaEst, np.transpose(invA)))
            CI = np.zeros((self._dim, 2))
            zBar = np.mean(self.z[:,initIndex:], axis=1)
            error_j = []
            for j in range(self._dim):
                lengthCI = np.sqrt(V[j, j] / (self._t * self._c * self._maxIter ** gamma)) * sts.norm.ppf(0.975)
                error_j.append(lengthCI)
                CI[j,:] = np.array([zBar[j] - lengthCI, zBar[j] + lengthCI])
            return zBar,np.max(error_j), V
        
    def _require_nb_samples(self,require_tol ,alpha_conf = 0.05):
        var_matrix = self.getPR()[2]
        max_var = np.max(np.diag(var_matrix))
        C_alpha = norm.ppf(1-alpha_conf /2)

        nb_sample = (max_var * C_alpha **2 ) /(self._c * self._t* require_tol**2)
        nb_sample = nb_sample ** (1/self._gamma)

        return nb_sample
        
          
    
    def H(self, x, z):
        m = z[0: self._dim]
        lam = z[-1]
        res = [0 for i in range(self._dim + 1)]
        res[-1] = self.l(x - m)
        res[0:self._dim] = lam * self.grad(x - m) - 1
        return np.array(res)
    
    def H2(self, x, rho, m):
        return rho - (np.sum(m) + self.l(x - m))
    
    def l(self, m):
        raise NotImplementedError()
        
    def grad(self, m):
        raise NotImplementedError()
    
### SA set up for exponential loss 
class SALoss1(SA):
    def __init__(self, X, c, gamma, K, t, init, epsilon, beta, alpha):
        self._alpha = alpha
        self._beta = beta
        super(SALoss1, self).__init__(X, c, gamma, K, t, init, epsilon)
    
    @property
    def beta(self):
        return self._beta
    
    @property
    def alpha(self):
        return self._alpha
    
    def l(self, m):
        m = np.array(m)
        exp_m = np.exp(m)
        sum_m = np.sum(m)
        s1, s2 = np.sum(self._beta * exp_m), np.exp(self._beta * sum_m)
        return (1 / (1 + self.alpha)) * (s1 + self.alpha * s2) - (self.dim + self._alpha)/ (1 + self._alpha)
        
    
    def grad(self, m):
        m = np.array(m)
        sum_m = np.sum(m)
        common_exp = np.exp(self._beta * sum_m * np.ones(self.dim))
        exp_m = np.exp(self._beta * m)
        return (self._beta / (1 + self._alpha)) * (exp_m + self._alpha * common_exp)
    

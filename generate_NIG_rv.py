import numpy as np
from numpy.random import default_rng
from scipy.linalg import cholesky
from scipy.stats import invgauss


class MNIGSampler:
    """
    Sampler for the multivariate Normal Inverse Gaussian (MNIG) distribution.

    Attributes:
        mu (ndarray[d]): Location vector \(\mu\).
        beta (ndarray[d]): Skewness vector \(\beta\).
        Sigma (ndarray[d, d]): Positive-definite covariance matrix \(\Sigma\).
        alpha1(float): Tail parameter (\(>0\)).
        delta (float): Scale parameter (\(>0\)).
        gamma (float): Mixing parameter \(\sqrt{\alpha^2 - \beta^\top \Sigma \beta}\).
        All of parameters alpha, beta and gamma is in the dictionarty kwargs.
        rng: numpy random number generator.
    """
    def __init__(self, mu, Sigma, alpha, beta, delta, seed=None):
        self.mu    = np.asarray(mu)
        self.Sigma = np.asarray(Sigma)
        self.beta  = np.asarray(beta)
        self.alpha = alpha
        self.delta = delta
        self.d = self.mu.shape[0]
        quad = float(self.beta.T @ self.Sigma @ self.beta)
        if alpha**2 <= quad:
            raise ValueError("α² must exceed βᵀΣβ")
        self.gamma = np.sqrt(alpha**2 - quad)
        self.Sigma_beta = self.Sigma @ self.beta

        
        self.Sigma_sqrt = cholesky(self.Sigma, lower=True)
        self._delta_sig_beta = self.delta * np.dot(self.Sigma, self.beta)
        self._mu_nig = self.mu- self._delta_sig_beta / self.gamma

        self.rng = default_rng(seed)

    def sample(self, n):
        # 1) Z ∼ IG(shape=δ, mean=δ/γ)
        # SciPy's invgauss uses the parameterization:
        #   invgauss(mu=mean, scale=shape)
        mu_ig    =  1 /(self.delta * self.gamma)
        scale_ig = self.delta**2
        # = gamma^{-2}. 
        Z = invgauss.rvs(mu = mu_ig,
                        scale =scale_ig,
                        size=n,
                        random_state=self.rng)
       
        Y = self.rng.standard_normal(size=(n, self.d))
        drift      = Z[:, None] * self.Sigma_beta
        noise      = (np.sqrt(Z)[:, None]
                    * (self.Sigma_sqrt @ Y.T).T)
        X = self._mu_nig + drift + noise
        
        return X




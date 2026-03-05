import sys
import numpy as np
path_pjt = "/Users/Nguye071/Documents/GitHub/Single-and-Multi-level-Fourier-RQMC-for-MSRM"
sys.path.insert(0, path_pjt)

from msra_loss import CommonLossFunctionAbs

from scipy.stats import norm

# ============================================================================
# SAA class for QPC loss
# ============================================================================
class MCLossFunction(CommonLossFunctionAbs):
    def __init__(self, distrib, alpha, c=None):
        """
        Initialize the SAA solver for QPC.

        Parameters:
        - distrib: disitribution of random loss vector X (Gaussian or NIG).
        - alpha: System-level coupling weight.
        - c: Optional inequality-constraint constant.
        """
        self._x = np.atleast_2d(distrib)
        self._alpha = alpha
        self.N, self.d = self._x.shape
        self._var_iter = []
        self._pending_var = None
        super(MCLossFunction, self).__init__(self.d, c)
    
    # Loss E[l(X - m)] 

    def shortfall_risk(self, m=None): 
        m = self._check_argument(m)
        x_minus_m = np.subtract(self._x, m)
        
        mean_sum_ = np.mean(x_minus_m.sum(axis=1))
        
        pos_part = np.maximum(x_minus_m, 0.)
        pos_part_squared = np.square(pos_part)
        mean_sum_2_ = np.mean(pos_part_squared.sum(axis=1))
        
        to_add = 0.
        for i in range(self.dim):
            for j in range(i + 1, self.dim):
                to_add += np.mean(np.multiply(pos_part[:, i], pos_part[:, j]))
                        
        return mean_sum_ + 0.5 * mean_sum_2_ + self._alpha * to_add
    
    # Loss gradient E[∇l(X - m)] 
    def shortfall_risk_jac(self, m):
        m = self._check_argument(m)
        x_minus_m = np.subtract(self._x, m)
        
        pos_part = np.maximum(x_minus_m, 0.)
        mean_pos_part = np.mean(pos_part, axis=0)
        
        dbl = []
        for i in range(self.dim):
            indic_i = np.where(pos_part[:, i] > 0, 1, 0)
            tmp = 0. 
            for j in range(self.dim):
                if i != j:
                    tmp += np.mean(np.multiply(indic_i, pos_part[:, j]))
                
            dbl.append(self._alpha * tmp)

        jac = mean_pos_part + 1. + dbl
        
        # Computing gradient variance part and return it in shortfall_risk_jac_var
        P = pos_part
        I = np.sign(P)
       
        full_outer = I[:, :, None] * P[:, None, :]
        dbl_samples = self._alpha * (full_outer.sum(axis=2) - np.diag(P * I))
        mean_dbl = dbl_samples.mean(axis=0)
        G = P - mean_pos_part[None, :] + dbl_samples - mean_dbl[None, :]

        if not np.all(np.isfinite(G)):
            G = np.nan_to_num(G, nan=0.0, posinf=0.0, neginf=0.0)

        cov = np.cov(G, rowvar=False, bias=False)

        self._pending_var = cov
        
        return jac
    
    # ============================================================================
    # Statistical error components
    # ============================================================================
    # Gradient-variance component of sandwich covariance V.
    def shortfall_risk_jac_var(self):
        var_comps = self._var_iter
    
        if not var_comps:
            raise RuntimeError("shortfall_risk_jac_var called before any gradient components were recorded")

        
        return var_comps[-1]
    # Inverse Hessian component of sandwich covariance V.
    def shortfall_risk_inv_hess(self,m):
        m = self._check_argument(m)
        x_minus_m = np.subtract(self._x, m)

        P = np.maximum(x_minus_m, 0.)
        I = np.sign(P)

        full_outer = I[:, :, None] * I[:, None, :]    

        H_samples = self._alpha * full_outer     

        idx = np.arange(self.d)
        H_samples[:, idx, idx] = I               


        H = H_samples.mean(axis=0)               
        if getattr(self,"N") ==1:
            return np.eye(self.d)
        
        H += 1e-10 * np.eye(self.d)
        return np.linalg.inv(H)
    
    # Maximum diagonal-based CI radius for the solution estimate.
    def statistical_error_sol_MC(self,m,alpha_conf = 0.05):
        var_jac = self.shortfall_risk_jac_var()
        inv_H = self.shortfall_risk_inv_hess(m)
        var_sol = inv_H @ var_jac @inv_H # (d,d)
    
        C_alpha = norm.ppf(1-alpha_conf /2)

        MC_stat_error = []

        for i in range(self.d):
            std_i = np.sqrt(var_sol[i,i])
            mc_err = C_alpha * std_i / np.sqrt(self.N) 
            MC_stat_error.append(mc_err)

        return np.max(MC_stat_error)
    
# ============================================================================
# SAA class for exponential loss
# ============================================================================
class MCLossFunction1(CommonLossFunctionAbs):
    def __init__(self, X, alpha, beta = 1,c=None):
        '''
        Parameters:
        - X: disitribution of random loss vector X (Gaussian or NIG).
        - alpha: System-level coupling weight.
        - beta: Risk-aversion coefficient.
        - c: Optional inequality-constraint constant.
        '''
        self._alpha = alpha
        self._beta = beta
        self._X = X
        self._N, self._d = self._X.shape
        self._var_iter = []
        self._pending_var = None
        super(MCLossFunction1, self).__init__(X.shape[1],c)
    
    # Loss E[l(X - m)] 
    def shortfall_risk(self, m = None): 
        m = self._check_argument(m)
        x_minus_m = np.subtract(self._X, m)
        exp_x_minus_m = np.exp( self._beta * x_minus_m)
        mean_sum1 = np.mean(np.sum(exp_x_minus_m, axis = 1))
        mean_sum2 = self._alpha * np.mean(np.exp( self._beta * np.sum(x_minus_m, axis = 1)))
        return (1 / (1 + self._alpha)) * (mean_sum1 + mean_sum2)
    
    # Loss gradient E[∇l(X - m)] 

    def shortfall_risk_jac(self, m):
        m = self._check_argument(m)
        x_minus_m = np.subtract(self._X, m)
        exp_ind = np.exp(self._beta * x_minus_m)
        exp_com = np.exp(self._beta * x_minus_m.sum(axis=1, keepdims=True))

        jac_samples = (self._beta / (1.0 + self._alpha)) * (exp_ind + self._alpha * exp_com)
        jac_mean = jac_samples.mean(axis=0)
        
        ## Computing the covariance part 
        jac_cov = np.cov(jac_samples, rowvar=False, ddof=1)
        self._pending_var = jac_cov

        return jac_mean
    
    # ============================================================================
    # Statistical error components
    # ============================================================================
    # Gradient-variance component of sandwich covariance V.
    def shortfall_risk_jac_var(self):
        var_comps = self._var_iter
    
        if not var_comps:
            raise RuntimeError("shortfall_risk_jac_var called before any gradient components were recorded")
        
        return var_comps[-1]
    
    # Inverse Hessian component of sandwich covariance V.
    def shortfall_risk_inv_hess(self,m):
        m = self._check_argument(m)
        x_minus_m = np.subtract(self._X, m)
        
        const_1 = (self._beta**2)/(1+self._alpha)
        hess = np.zeros((self._N, self._d, self._d), dtype = float)
        
        exp_ind = np.exp(self._beta * x_minus_m)             
        exp_com = np.exp(self._beta * x_minus_m.sum(axis=1, keepdims=True))
        exp_com = np.asarray(exp_com).reshape(self._N)       

        hess = np.empty((self._N, self._d, self._d), dtype=float)
        hess[:] = self._alpha * exp_com[:, None, None]     

        idx = np.arange(self._d)
        hess[:, idx, idx] += exp_ind                      

        hess *= const_1

        H_bar = hess.mean(axis=0)                        
        H_bar_inv = np.linalg.inv(H_bar)
        return H_bar_inv
    
    # Maximum diagonal-based CI radius for the solution estimate.
    def statistical_error_sol_MC(self,m,alpha_conf=0.05):
        var_jac = self.shortfall_risk_jac_var()
        inv_H = self.shortfall_risk_inv_hess(m)
        var_sol = inv_H @ var_jac @inv_H
    
        C_alpha = norm.ppf(1-alpha_conf /2)

        MC_stat_error = []

        for i in range(self._d):
            std_i = np.sqrt(var_sol[i,i])
            mc_err = C_alpha * std_i / np.sqrt(self._N) 
            MC_stat_error.append(mc_err)

        return np.max(MC_stat_error)
    

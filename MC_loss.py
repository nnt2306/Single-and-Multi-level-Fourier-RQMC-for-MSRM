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

    def loss_derivative_samples(self, m):
        """Loss and positive-gradient samples: (N,) and (N, d)."""
        m = self._check_argument(m)
        y = self._x - m

        positive = np.maximum(y, 0.0)
        indicator = (y > 0.0).astype(float)

        positive_sum = positive.sum(axis=1)
        squared_sum = (positive**2).sum(axis=1)

        cross_term = 0.5 * (
            positive_sum**2 - squared_sum
        )

        loss_samples = (
            y.sum(axis=1)
            + 0.5 * squared_sum
            + self._alpha * cross_term
        )

        other_positive_sum = (
            positive_sum[:, None] - positive
        )


        gradient_samples = (
            1.0
            + positive
            + self._alpha * indicator * other_positive_sum
        )

        return loss_samples, gradient_samples


    def shortfall_risk(self, m=None):
        loss_samples, _ = self.loss_derivative_samples(m)
        return float(loss_samples.mean())

    
    # Loss gradient E[∇l(X - m)] 
    def shortfall_risk_jac(self, m):
        _, gradient_samples = self.loss_derivative_samples(m)
        return gradient_samples.mean(axis=0)

    
    # ============================================================================
    # Statistical error components
    # ============================================================================
    def mean_loss_hessian(self, m):
        """Mean QPC Hessian, including the diagonal boundary correction."""
        m = self._check_argument(m)
        y = self._x - m
        n_samples, d = y.shape

        positive = np.maximum(y, 0.0)
        indicator = (y > 0.0).astype(float)

        hessian = (
            self._alpha
            * (indicator.T @ indicator)
            / n_samples
        )

        diagonal = indicator.mean(axis=0)

        if self._alpha:
            if n_samples < 2:
                raise ValueError(
                    "At least two observations are required."
                )

            scale = self._x.std(axis=0, ddof=1)
            bandwidth = 0.5 * scale * n_samples**(-0.2)
            ## Booundary correction term using finite difference
            if np.any(~np.isfinite(bandwidth)) or np.any(bandwidth <= 0):
                raise ValueError(
                    "Boundary estimation requires positive sample "
                    "standard deviations in every component."
                )

            other_positive_sum = (
                positive.sum(axis=1, keepdims=True) - positive
            )


            boundary_weight = (
                (y > -bandwidth)
                & (y <= bandwidth)
            ).astype(float) / (2.0 * bandwidth)

            diagonal += self._alpha * np.mean(
                other_positive_sum * boundary_weight,
                axis=0,
            )


        np.fill_diagonal(hessian, diagonal)

        return hessian


    def gradient_oracle(self, loss_samples, gradient_samples, lam):
        """One full oracle vector per observation: shape (N, d+1)."""
        n_samples, d = gradient_samples.shape

        oracle_samples = np.empty(
            (n_samples, d + 1), dtype=float
        )

        oracle_samples[:, :d] = 1.0 - lam * gradient_samples
        oracle_samples[:, -1] = loss_samples - self.c

        return oracle_samples


    def jacobian_oracle(self, m, lam, mean_gradient):
        """Mean oracle Jacobian with respect to (m, lambda)."""
        d = len(mean_gradient)

        jacobian = np.zeros(
            (d + 1, d + 1), dtype=float
        )

        jacobian[:d, :d] = lam * self.mean_loss_hessian(m)
        jacobian[:d, -1] = -mean_gradient
        jacobian[-1, :d] = -mean_gradient

        return jacobian


    def solution_covariance(self, m, lam):
        """Full covariance for (m, lambda): shape (d+1, d+1)."""
        m = self._check_argument(m)

        loss_samples, gradient_samples = (
            self.loss_derivative_samples(m)
        )
        n_samples = gradient_samples.shape[0]

        if n_samples < 2:
            raise ValueError(
                "At least two observations are required."
            )

        oracle_samples = self.gradient_oracle(
            loss_samples, gradient_samples, lam
        )

        oracle_covariance = np.cov(
            oracle_samples,
            rowvar=False,
            ddof=1,
        )

        mean_jacobian = self.jacobian_oracle(
            m,
            lam,
            gradient_samples.mean(axis=0),
        )

        left = np.linalg.solve(
            mean_jacobian, oracle_covariance
        )
        sandwich = np.linalg.solve(
            mean_jacobian, left.T
        ).T

        return sandwich / n_samples


    def stat_err_full(self, m, lam, alpha_conf=0.05):
        """Marginal confidence half-widths for allocation components."""
        covariance = self.solution_covariance(m, lam)

        critical_value = norm.ppf(
            1.0 - alpha_conf / 2.0
        )


        errors = critical_value * np.sqrt(
            np.diag(covariance)
        )

        return float(errors.max())
    

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
    def loss_derivative_samples(self, m):
        m = self._check_argument(m)
        y = self._X - m

        exp_ind = np.exp(self._beta * y)
        exp_common = np.exp(
            self._beta * y.sum(axis=1)
        )

        scale = 1.0 / (1.0 + self._alpha)

        loss_samples = scale * (
            exp_ind.sum(axis=1)
            + self._alpha * exp_common
        )


        gradient_samples = self._beta * scale * (
            exp_ind
            + self._alpha * exp_common[:, None]
        )

        return loss_samples, gradient_samples


    def shortfall_risk(self, m=None):
        loss_samples, _ = self.loss_derivative_samples(m)
        return float(loss_samples.mean())

    
    # Loss gradient E[∇l(X - m)] 

    def shortfall_risk_jac(self, m):
        _, gradient_samples = self.loss_derivative_samples(m)

        self._pending_var = np.atleast_2d(
        ## Computing the covariance part 
            np.cov(gradient_samples, rowvar=False, ddof=1)
        )

        return gradient_samples.mean(axis=0)

     # ============================================================================
    # Statistical error components
    # ============================================================================

    def mean_loss_hessian(self, m):
        """Mean Hessian only; no (N, d, d) array is needed."""
        m = self._check_argument(m)
        y = self._X - m

        mean_exp_ind = np.exp(
            self._beta * y
        ).mean(axis=0)

        mean_exp_common = np.exp(
            self._beta * y.sum(axis=1)
        ).mean()

        hessian = np.diag(mean_exp_ind)

        hessian += (
            self._alpha
            * mean_exp_common
            * np.ones((self._d, self._d))
        )

        return (
            self._beta**2 / (1.0 + self._alpha)
        ) * hessian


    def gradient_oracle(self, loss_samples, gradient_samples, lam):
        """One oracle vector per observation: shape (N, d+1)."""
        n_samples, d = gradient_samples.shape

        oracle_samples = np.empty(
            (n_samples, d + 1), dtype=float
        )

        oracle_samples[:, :d] = 1.0 - lam * gradient_samples
        oracle_samples[:, -1] = loss_samples - self.c

        return oracle_samples


    def jacobian_oracle(self, m, lam, mean_gradient):
        """Mean oracle Jacobian with respect to (m, lambda)."""
        jacobian = np.zeros(
            (self._d + 1, self._d + 1), dtype=float
        )

        jacobian[:self._d, :self._d] = (
            lam * self.mean_loss_hessian(m)
        )
        jacobian[:self._d, -1] = -mean_gradient
        jacobian[-1, :self._d] = -mean_gradient

        return jacobian


    def solution_covariance(self, m, lam):
        """Full covariance for (m, lambda): shape (d+1, d+1)."""
        m = self._check_argument(m)

        loss_samples, gradient_samples = (
            self.loss_derivative_samples(m)
        )
        n_samples = gradient_samples.shape[0]

        if n_samples < 2:
            raise ValueError("At least two observations are required.")

        oracle_samples = self.gradient_oracle(
            loss_samples, gradient_samples, lam
        )

        oracle_covariance = np.cov(
            oracle_samples,
            rowvar=False,
            ddof=1,
        )

        mean_jacobian = self.jacobian_oracle(
            m,
            lam,
            gradient_samples.mean(axis=0),
        )

        left = np.linalg.solve(
            mean_jacobian, oracle_covariance
        )
        sandwich = np.linalg.solve(
            mean_jacobian, left.T
        ).T

        return sandwich / n_samples


    def stat_err_full(self, m, lam, alpha_conf=0.05):
        """Marginal confidence half-widths for allocation components."""
        covariance = self.solution_covariance(m, lam)

        critical_value = norm.ppf(
            1.0 - alpha_conf / 2.0
        )

        errors = critical_value * np.sqrt(
            np.diag(covariance)
        )

        return float(errors.max())

   
   


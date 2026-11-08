import sys
import numpy as np
## Exponential Loss function 
# Closed form in 2D
class ClosedForm2D:
    def __init__(self, v_sigma, beta, alpha):

        if not isinstance(v_sigma, np.ndarray):
            v_sigma = np.array(v_sigma)

        self._v_sigma = v_sigma
        self._beta = beta
        self._alpha= alpha
        self._sigma_x = np.sqrt(v_sigma[0, 0])
        self._sigma_y = np.sqrt(v_sigma[1, 1])
        self._rho = self._v_sigma[0, 1] / (self._sigma_x *self._sigma_y)

    def compute(self):
        """Compute the two-dimensional m and lambda parameters."""
       

        # Independent one-dimensional contribution for each component.
        base_m = 0.5 * self._beta * np.diag(self._v_sigma)[:2]

        if self._alpha == 0.0:
            return base_m

        # Interaction contribution.
        correlation_exponent = (
            self._rho
            * self._beta**2
            * self._sigma_x
            * self._sigma_y
        )
        interaction_exponential = np.exp(correlation_exponent)
        scaled_interaction = self._alpha * interaction_exponential

        square_root_term = np.sqrt(
            1.0
            + self._alpha
            * (self._alpha + 2.0)
            * interaction_exponential
        )

        logarithm_denominator = square_root_term - 1.0
        logarithm_argument = (
            scaled_interaction / logarithm_denominator
        )
        interaction_shift = np.log(logarithm_argument) / self._beta

        ## m solution
        m = base_m + interaction_shift 

        # Lambda contribution.
        total_m = np.sum(m)

        marginal_exponential_1 = np.exp(base_m[0] - self._beta * m[0])
        marginal_12 = 0.5 * self._beta**2 * (self._sigma_x **2 + self._sigma_y**2)
        joint_m_12= self._beta * total_m

        total_m_12 = correlation_exponent + marginal_12 - joint_m_12
        exp_m_12 = self._alpha * np.exp(total_m_12)

        lambda_denominator = (
            marginal_exponential_1
            + exp_m_12
        )
        lam = (1.0 + self._alpha) / lambda_denominator
        #logging.info(f"Value of lam is {lam}")

        return np.array([m[0], m[1], lam], dtype=float)


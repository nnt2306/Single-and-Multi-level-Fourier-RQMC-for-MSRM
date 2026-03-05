import sys
import numpy as np
###### Exponential Loss function 
## Closed form in 2D
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
        base_result = np.array([
            0.5 *self._beta* self._v_sigma[0, 0],
            0.5 *self._beta* self._v_sigma[1, 1]
        ])

        if self._alpha == 0:
            return base_result

        exp_term = np.exp(self._rho *self._beta**2 * self._sigma_x *self._sigma_y)
        numerator = self._alpha* exp_term
        denominator = -1 + np.sqrt(1 + self._alpha * (self._alpha + 2) * exp_term)
        src_term = np.log(numerator / denominator)

        return base_result + (src_term / self._beta) * np.ones(2)

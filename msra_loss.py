
import numpy as np


class CommonLossFunctionAbs(object):
    def __init__(self, dim, c=None):
        self._dim= dim
        self._c= c

    @property
    def dim(self):
        return self._dim

    @property
    def c(self):
        if self._c is None:
            raise AttributeError("The value of c is unset")
        return self._c

    @c.setter
    def c(self, c):
        self._c= c

    def _check_argument(self, m):
        if m is None:
            m = np.zeros((self._dim))
        else:
            if m.shape != (self._dim,):
                raise ValueError("m must be of shape (%i,). Given: %s." % (self._dim, m.shape))
        return m

    def objective(self, m):
        return np.sum(m)

    def objective_jac(self, m):
        return np.ones((self._dim))

    def ineq_constraint(self, m):
        return self.c - self.shortfall_risk(m)

    def ineq_constraint_jac(self, m):
        return self.shortfall_risk_jac(m)

    # These methods must be implemented by subclasses.
    def shortfall_risk(self, m=None):
        raise NotImplementedError("shortfall_risk must be implemented in the subclass")

    def shortfall_risk_jac(self, m):
        raise NotImplementedError("shortfall_risk_jac must be implemented in the subclass")
    

   

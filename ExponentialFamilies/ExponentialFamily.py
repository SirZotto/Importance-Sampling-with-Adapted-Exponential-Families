import numpy as np
import cvxpy as cp

class ExponentialFamily():
    
    def __init__(self, q_0, T,z ,A):
        """Define the components of an exponential-family model.

        Args:
        q_0 (function/method): Takes values `x` (which may be NumPy array-like)
                               such that `q_0(x)` returns a float. Represents the base density.
        T (array-like function/method): Takes values `x` (which may be NumPy
                                        array-like) such that `T(x)` returns a float or NumPy array-like.
                                        Represents the sufficient statistic.
        z (array-like function/method): Takes values `x` (which may be NumPy
                                        array-like) such that `z(x)` returns a float or NumPy array-like.
                                        Represents the natural parameter. `z` must not violate any
                                        measurability rules (see README.md).
        A (function/method): Takes `z(x)` as input such that `A(z)` returns a
                             float. Represents the log-partition function.
        """

        self.q_0 = q_0
        self.T = T
        self.z =z
        self.A = A
     
    def getBaseDensity(self):
        return self.q_0
    
    def getSufficientStatistic(self):
        return self.T
    
    def getNaturalParameter(self):
        return self.z
    
    def getLogPartition(self):
        return self.A

    def sample(self, size):
        """generate size-many samples according to the exponential family as pdf
        """
        raise NotImplementedError("Each ExponentialFamily subclass must implement sample().")
    
    def operation_zT(self):
        """the operation used to calculate the z^T T. This operation may be just the normal skalar product.
        """
        raise NotImplementedError("Each ExponentialFamily subclass must implement operation_zT.")
    
    def optimizeForIS(self,
        outterFunction_f, 
        underlyingDensity_p, 
        samplesFromUnderlying_p = None,
        startingAlpha =None, 
        bounds=None, 
        cvxpy_solver=cp.SCS, 
        cvxpy_verbose=False,
        cvxpy_max_iters=20000, 
        cvxpy_eps=1e-5, 
        cvxpy_acceleration_lookback=10, 
        cvxpy_scale=1.0):
        """Optimzes the the natrual parameter according to the procedure described in the README.md. Keyword: variance Reduction.
        """
        raise NotImplementedError("Each ExponentialFamily subclass must implement optimizeForIs(...).")

    def __call__(self, x):
        """Calls the defined exponential-family        

        Args:
            x (array-like/float): `x` must match the requirement from the components in the __init__.
        """
        q_0Eval = self.q_0(x)
        TEval = self.T(x)
        zEval = self.z(x)
        AEval = self.A(zEval)
        
        zTEval = self.operation_zT(zEval,TEval)
        
        return q_0Eval * np.exp(zTEval -AEval)
        
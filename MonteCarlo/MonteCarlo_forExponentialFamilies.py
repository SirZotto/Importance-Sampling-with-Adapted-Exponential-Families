from scipy.stats import norm

def ci_value(confidence_level: float) -> float:
    """Gives the value for a two sided symmetric confidence interval.
    Example:
    ci_value(0.95)  # 1.959963984540054  ≈ 1.96
    """
    return norm.ppf((1 + confidence_level) / 2)

class MonteCarlo_forExponentialFamilies:
    """Provides a tool to generate a MonteCarlo/Importance sampling approximation using the ExponentialFamily via
    several methods usually used within the context of Monte Carlo.    
       
    """
    
    def __init__(self, exponentialFamily_q, outerFunction_f, underlyingDensity_p = None):
        """Takes an exponential-family proposal q, an outer function f, and an
    optional underlying density p.
    Depending on p, the estimator is:

    - Simple Monte Carlo (if p is None):

        (1 / M) * sum_{m=1..M} f(X_m),  where X_m ~ q

    - Importance-sampling Monte Carlo (if p is not None):

        (1 / M) * sum_{m=1..M} [p(X_m) / q(X_m)] * f(X_m),
        where X_m ~ q

      This requires q(X_m) > 0 whenever p(X_m) * f(X_m) contributes.
        
        Args:
            exponentialFamily (ExponentialFamily): must be of type ExponentialFamily to allow running all underlying methods
            outerFunction_f (callable):  must take same input  as exponentialFamily __call__
            underlyingDensity (callable): must take same input  as exponentialFamily __call__
        """
        
        self.underlyingDensity = underlyingDensity_p
        self.outerFunction_f = outerFunction_f
        self. exponentialFamily = exponentialFamily_q
        

    def estimate(self,sampleAmount, seed):
        """estimates the Monte Carlo approximation. D
        Depending on p, the estimator is:

        - Simple Monte Carlo (if p is None):

            (1 / M) * sum_{m=1..M} f(X_m),  where X_m ~ q

        - Importance-sampling Monte Carlo (if p is not None):

            (1 / M) * sum_{m=1..M} [p(X_m) / q(X_m)] * f(X_m),
            where X_m ~ q

        Args:
            sampleAmount (int): Equivalent to M, the amount of samples used to generate the Monte Carlo approximation.
            seed (int): seed for recreation purposes 
        """
        raise NotImplementedError("Must be implemented")
    
    def confidenceInterval(self,sampleAmount, boundaryLevel, seed):
        """Generates only the confidence interval of the MonteCarlo approximation 
        according to the boundary level (f.e. 95% )

        Args:
            sampleAmount (int): Equivalent to M, the amount of samples used to generate the Monte Carlo approximation.
            boundryLevel (float):must be a value between in (0,1) representing the desired percentage level of the confidence interval.
                                Example: 0.95 represents a 95% confidence interval.
           seed (int): seed for recreation purposes 
        """
        raise NotImplementedError("Must be implemented")
        
        
    def estimateWithConfidenceInterval(self,sampleAmount, boundaryLevel, seed):
        """Generates only the confidence interval of the MonteCarlo approximation 
        according to the boundary level (f.e. 95% )

        Args:
            sampleAmount (int): Equivalent to M, the amount of samples used to generate the Monte Carlo approximation.
            boundryLevel (float):must be a value between in (0,1) representing the desired percentage level of the confidence interval.
                                Example: 0.95 represents a 95% confidence interval.
           seed (int): seed for recreation purposes 
        """
        raise NotImplementedError("Must be implemented")
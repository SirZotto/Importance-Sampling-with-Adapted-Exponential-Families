import numpy as np
import sys
from pathlib import Path
from typing import override
import warnings
import cvxpy as cp
sys.path.append(str(Path(__file__).resolve().parents[1]))

from NaturalParameterGenerator.NormalDistributionNaturalParameter import NormalDistributionNaturalParameter
from ExponentialFamilies.ExponentialFamily import ExponentialFamily


class NormalDistributionFamily(ExponentialFamily):
    """
    Generates the Normal Distribution with the context of Exponential families.

    Args:
        ExponentialFamily (_type_): _description_
    """

    @override
    def __init__(
        self,
        d,
        S,
        z=None,
        zType_function=None,
        zCombination_function=None,
        zType_parameters=(),
        zType_keyword_parameters=None,
    ):
        if not isinstance(d, int) or d <= 0:
            raise ValueError("d must be a positive integer.")
        if not isinstance(S, int) or S <= 0:
            raise ValueError("S must be a positive integer.")

        self.d = d
        self.S = S

        if z is None:
            z = NormalDistributionNaturalParameter(
                d=d,
                S=S,
                type_function=(
                    "id" if zType_function is None else zType_function
                ),
                combination_function=(
                    "mean_out_of_typ"
                    if zCombination_function is None
                    else zCombination_function
                ),
                type_parameters=zType_parameters,
                type_keyword_parameters=zType_keyword_parameters,
            )
        elif (
            zType_function is not None
            or zCombination_function is not None
            or zType_parameters
            or zType_keyword_parameters is not None
        ):
            raise ValueError(
                "Specify either z or its construction settings, not both."
            )

        if not callable(z):
            raise TypeError("z must be callable on a history of observations.")

        self._base_z = z
        self.alpha = np.ones((S, d), dtype=float)

        q_0 = lambda x: (2.0 * np.pi) ** (-S * d / 2.0)
        T = lambda x: (x, np.outer(x, x))

        super().__init__( q_0=q_0,T=T,z=self._effective_z,A=self._A) 
        
    def __call__(self, X):
        return float(np.exp(self.log_density(X)))    
    
    
    def sample(self, size, rng=None):
        if not isinstance(size, int) or size < 0:
            raise ValueError("size must be a non-negative integer.")

        rng = np.random.default_rng(rng)
        samples = np.empty((size, self.S, self.d), dtype=float)

        for i in range(size):
            history = []

            for s in range(self.S):
                lam, Lam = self.z(history)

                precision = -2.0 * Lam
                covariance = np.linalg.solve(
                    precision,
                    np.eye(self.d),
                )
                mean = covariance @ lam

                x_s = rng.multivariate_normal(mean, covariance)
                samples[i, s] = x_s
                history.append(x_s)

        return samples

    def _effective_z(self, history):
        lam, Lam = self._base_z(history)
        step = len(history)

        if step >= self.S:
            raise ValueError("History is longer than the sequence allows.")

        lam = np.asarray(lam, dtype=float)
        Lam = np.asarray(Lam, dtype=float)

        if lam.shape != (self.d,) or Lam.shape != (self.d, self.d):
            raise ValueError(
                "z must return lambda with shape (d,) "
                "and Lambda with shape (d, d)."
            )

        return self.alpha[step] * lam, Lam 
    
    def operation_zT(self, natural_parameter, sufficient_statistic):
        lam, Lam = natural_parameter
        x, xxT = sufficient_statistic

        return float(np.dot(lam, x) + np.vdot(Lam, xxT))
    
    def log_density(self, X):
        X = np.asarray(X, dtype=float)

        if X.shape != (self.S, self.d):
            raise ValueError(
                f"X must have shape ({self.S}, {self.d})."
            )
        if not np.all(np.isfinite(X)):
            raise ValueError("X must contain only finite values.")

        log_q = -0.5 * self.S * self.d * np.log(2.0 * np.pi)

        for s in range(self.S):
            eta_s = self.z(X[:s])
            T_s = self.T(X[s])

            log_q += (
                self.operation_zT(eta_s, T_s)
                - self.A(eta_s)
            )

        return float(log_q)
        
    @staticmethod
    def _A(z, history=None, alpha=None):
        """
        Normal log-partition function for one conditional distribution.

        Args:
            z: Either a callable natural-parameter rule or an evaluated
            (lambda, Lambda) pair.
            history: Previous observations; required when z is callable.
            alpha: Optional length-d vector that scales lambda coordinate-wise.
                May be a NumPy array or a CVXPY expression.

        Returns:
            A number for numeric inputs, or a CVXPY expression when alpha
            is a CVXPY expression.
        """
        if callable(z):
            if history is None:
                raise ValueError("history is required when z is callable.")
            lam, Lam = z(history)
        else:
            if history is not None:
                raise ValueError("Pass history only when z is callable.")
            lam, Lam = z

        lam = np.asarray(lam, dtype=float)
        Lam = np.asarray(Lam, dtype=float)

        if lam.ndim != 1 or Lam.shape != (lam.size, lam.size):
            raise ValueError("Expected lambda with shape (d,) and Lambda with shape (d, d).")
        if not np.all(np.isfinite(lam)) or not np.all(np.isfinite(Lam)):
            raise ValueError("Natural parameters must be finite.")
        if not np.allclose(Lam, Lam.T):
            raise ValueError("Lambda must be symmetric.")

        K = -2.0 * Lam
        try:
            L = np.linalg.cholesky(K)
        except np.linalg.LinAlgError as error:
            raise ValueError("Lambda must be negative definite.") from error

        half_logdet = np.sum(np.log(np.diag(L)))

        if isinstance(alpha, cp.Expression):
            if alpha.shape != lam.shape:
                raise ValueError(f"alpha must have shape {lam.shape}.")
            K_inv = np.linalg.solve(K, np.eye(lam.size))
            scaled_lam = cp.multiply(alpha, lam)
            return 0.5 * cp.quad_form(scaled_lam, cp.psd_wrap(K_inv)) - half_logdet

        if alpha is not None:
            alpha = np.asarray(alpha, dtype=float)
            if alpha.shape != lam.shape:
                raise ValueError(f"alpha must have shape {lam.shape}.")
            lam = alpha * lam

        return 0.5 * lam @ np.linalg.solve(K, lam) - half_logdet



    def optimizeForIS(
        self,
        outterFunction_f,
        underlyingDensity_p,
        samplesFromUnderlying_p=None,
        startingAlpha=None,
        bounds=None,
        cvxpy_solver=cp.SCS,
        cvxpy_verbose=False,
        cvxpy_max_iters=20000,
        cvxpy_eps=1e-5,
        cvxpy_acceleration_lookback=10,
        cvxpy_scale=1.0,
    ):
        if samplesFromUnderlying_p is None:
            raise ValueError(
                "samplesFromUnderlying_p is required: "
                "the density function alone cannot generate samples."
            )

        samples = np.asarray(samplesFromUnderlying_p, dtype=float)
        if samples.ndim != 3 or samples.shape[1:] != (self.S, self.d):
            raise ValueError(
                f"Samples must have shape (M, {self.S}, {self.d})."
            )
        if len(samples) == 0 or not np.all(np.isfinite(samples)):
            raise ValueError("Provide at least one finite sample.")

        alpha = cp.Variable(self.S * self.d)

        if startingAlpha is not None:
            initial = np.asarray(startingAlpha, dtype=float)
            if initial.size != self.S * self.d:
                raise ValueError(
                    f"startingAlpha must contain {self.S * self.d} values."
                )
            alpha.value = initial.reshape(self.S * self.d)

        constraints = [] if bounds is None else list(bounds(alpha))
        terms = []

        for X in samples:
            p_value = float(underlyingDensity_p(X))
            f_value = float(outterFunction_f(X))

            if (
                not np.isfinite(p_value)
                or p_value < 0
                or not np.isfinite(f_value)
            ):
                raise ValueError("p and f must return finite values, with p >= 0.")

            # Such a sample contributes zero to the second-moment objective.
            if p_value == 0 or f_value == 0:
                continue

            # Samples are drawn from p, so the Monte Carlo integrand is
            # p(X) * f(X)^2 / q_alpha(X).
            term = (
                np.log(p_value)
                + 2.0 * np.log(abs(f_value))
                + 0.5 * self.S * self.d * np.log(2.0 * np.pi)
            )

            for s in range(self.S):
                lam, Lam = self._base_z(X[:s])
                lam = np.asarray(lam, dtype=float)
                Lam = np.asarray(Lam, dtype=float)

                alpha_s = alpha[s * self.d:(s + 1) * self.d]

                term += self._A((lam, Lam), alpha=alpha_s)
                term -= cp.sum(
                    cp.multiply(
                        cp.multiply(alpha_s, lam),
                        X[s],
                    )
                )
                term -= np.vdot(Lam, np.outer(X[s], X[s]))

            terms.append(term)

        if not terms:
            raise ValueError(
                "All samples have zero contribution to the objective."
            )

        # Minimizing log(sum(exp(terms))) has the same minimizer as
        # minimizing the positive second-moment estimate.
        problem = cp.Problem(
            cp.Minimize(cp.log_sum_exp(cp.hstack(terms))),
            constraints,
        )

        solve_options = {
            "solver": cvxpy_solver,
            "verbose": cvxpy_verbose,
        }
        if cvxpy_solver == cp.SCS:
            solve_options.update(
                max_iters=cvxpy_max_iters,
                eps=cvxpy_eps,
                acceleration_lookback=cvxpy_acceleration_lookback,
                scale=cvxpy_scale,
            )

        problem.solve(**solve_options)

        if alpha.value is None:
            raise RuntimeError(
                f"Optimization failed with status {problem.status!r}."
            )

        optimized_alpha = np.asarray(alpha.value).reshape(
            self.S, self.d
        )
        if not np.all(np.isfinite(optimized_alpha)):
            raise RuntimeError("Optimization returned non-finite alpha.")

        self.alpha = optimized_alpha.copy()
        self.optimization_problem = problem

        return self.alpha.copy()
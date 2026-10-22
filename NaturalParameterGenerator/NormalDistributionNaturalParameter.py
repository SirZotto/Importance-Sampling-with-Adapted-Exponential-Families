import numpy as np
import sys
from pathlib import Path
from typing import override
sys.path.append(str(Path(__file__).resolve().parents[1]))


from NaturalParameterGenerator.NaturalParameter import NaturalParameter


class NormalDistributionNaturalParameter(NaturalParameter):
    """
    Generate history-dependent natural parameters for conditional
    multivariate Normal distributions.
    """

    def __init__(
        self,
        d,
        S,
        mu=None,
        SIG=None,
        type_function="id",
        combination_function="mean_out_of_typ",
        type_parameters=(),
        type_keyword_parameters=None,
    ):
        if not isinstance(d, int) or d <= 0:
            raise ValueError("d must be a positive integer.")

        if not isinstance(S, int) or S <= 0:
            raise ValueError("S must be a positive integer.")

        self.d = d
        self.S = S

        self.mu = (
            np.ones(d, dtype=float)
            if mu is None
            else np.asarray(mu, dtype=float)
        )
        if self.mu.shape != (d,):
            raise ValueError(f"mu must have shape ({d},).")

        self.SIG = (
            np.eye(d, dtype=float)
            if SIG is None
            else np.asarray(SIG, dtype=float)
        )
        if self.SIG.shape != (d, d):
            raise ValueError(f"SIG must have shape ({d}, {d}).")

        if not np.allclose(self.SIG, self.SIG.T):
            raise ValueError("SIG must be symmetric.")

        try:
            np.linalg.cholesky(self.SIG)
        except np.linalg.LinAlgError as error:
            raise ValueError(
                "SIG must be positive definite."
            ) from error

        self.SIG_inv = np.linalg.inv(self.SIG)

        # Parameters forwarded to TypeFunction when history is combined.
        self.type_parameters = tuple(type_parameters)
        self.type_keyword_parameters = (
            {}
            if type_keyword_parameters is None
            else dict(type_keyword_parameters)
        )

        super().__init__(
            type_function=type_function,
            combination_function=combination_function,
        )

    def generate(self):
        """
        Create the Normal natural-parameter rule.

        The returned callable accepts a history with shape
        (number_of_previous_samples, d).

        - For empty history, it returns the initial natural parameter:
          (SIG_inv @ mu, -0.5 * SIG_inv).

        - Otherwise, it constructs lambda from the configured TypeFunction
          and CombinationFunction, while Lambda remains fixed.
        """
        lambda_initial = self.SIG_inv @ self.mu
        Lambda = -0.5 * self.SIG_inv

        def z(history):
            history = np.asarray(history, dtype=float)

            if history.size == 0:
                return lambda_initial.copy(), Lambda.copy()

            if history.ndim != 2 or history.shape[1] != self.d:
                raise ValueError(
                    f"history must have shape (number_of_previous_samples, "
                    f"{self.d})."
                )

            if history.shape[0] >= self.S:
                raise ValueError(
                    f"history can contain at most {self.S - 1} previous "
                    "samples."
                )

            lambda_s = self.combinationFunction(
                history,
                self.typeFunction,
                *self.type_parameters,
                **self.type_keyword_parameters,
            )
            lambda_s = np.asarray(lambda_s, dtype=float)

            if lambda_s.shape != (self.d,):
                raise ValueError(
                    f"The combined type function must return a vector with "
                    f"shape ({self.d},), but returned {lambda_s.shape}."
                )

            return lambda_s, Lambda.copy()

        return z
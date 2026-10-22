import numpy as np
from scipy.stats import expon, norm


class TypeFunction:
    """
    A callable wrapper for transformations used to construct natural parameters.

    Examples:
        relu = TypeFunction("ReLU")
        relu(-2.0)

        leaky_relu = TypeFunction("LReLU")
        leaky_relu(-2.0, 0.1)

        custom = TypeFunction(my_custom_function)
        custom(x, parameter1, parameter2)
    """

    def __init__(self, type_function):
        """
        Initialise a type transformation.

        A type transformation can be selected by name from the built-in
        transformations or supplied directly as a callable. A supplied callable
        must accept an observation as its first argument. Any further positional
        or keyword arguments are forwarded when this TypeFunction is called.

        Args:
            type_function (str or callable): Either the name of a built-in
                transformation (`"ReLU"`, `"LReLU"`, `"shiftReLU"`, `"id"`,
                `"constant"`, `"cappedReLU"`, `"stdNormal"`, `"abs"`, or
                `"stdExp_dist"`), or a callable such as a function, lambda,
                or bound method.

        Raises:
            ValueError: If `type_function` is a string that does not identify
                one of the supported built-in transformations.
            TypeError: If `type_function` is neither a string nor a callable.
        """
        if isinstance(type_function, str):
            available_type_functions = {
                "ReLU": self.ReLU,
                "LReLU": self.LReLU,
                "shiftReLU": self.shiftReLU,
                "id": self.identity,
                "constant": self.constant,
                "cappedReLU": self.cappedReLU,
                "stdNormal": self.standard_normal,
                "abs": np.abs,
                "stdExp_dist": self.standard_exponential,
            }

            try:
                self.typ = available_type_functions[type_function]
            except KeyError as error:
                valid_names = ", ".join(available_type_functions)
                raise ValueError(
                    f"Unknown type function {type_function!r}. "
                    f"Choose one of: {valid_names}."
                ) from error

        elif callable(type_function):
            # A supplied function/method becomes the transformation directly.
            self.typ = type_function

        else:
            raise TypeError(
                "type_function must be a supported string or a callable "
                "(function, lambda, or method)."
            )

    def __call__(self, x, *parameters, **keyword_parameters):
        """
        Evaluate the selected transformation/ type function.

        Extra positional and keyword parameters are passed straight to self.typ,
        so future transformations may have any parameter signature.
        """
        return self.typ(x, *parameters, **keyword_parameters)

    @staticmethod
    def ReLU(x):
        return np.maximum(0, x)

    @staticmethod
    def LReLU(x, beta):
        return np.where(np.asarray(x) < 0, np.asarray(x) * beta, x)

    @staticmethod
    def shiftReLU(x, beta):
        return np.maximum(x, beta)

    @staticmethod
    def identity(x):
        return x

    @staticmethod
    def constant(x, value):
        x = np.asarray(x)
        return np.full_like(
            x,
            fill_value=value,
            dtype=np.result_type(x.dtype, np.asarray(value).dtype),
        )

    @staticmethod
    def cappedReLU(x, beta):
        return np.minimum(np.maximum(0, x), beta)

    @staticmethod
    def standard_normal(x, beta):
        x = np.asarray(x)
        return np.where(np.abs(x) <= beta, norm.pdf(x), norm.pdf(beta))

    @staticmethod
    def standard_exponential(x, beta):
        x = np.asarray(x)
        return np.where(
            np.abs(x) <= beta,
            expon(scale=1).pdf(x),
            expon(scale=1).pdf(beta),
        )
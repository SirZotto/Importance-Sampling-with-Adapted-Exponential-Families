import numpy as np
###this class was coded my ai under my supervision, the orginal code is from my bachelors thesis thus the ai upgraded the code there in this file

class CombinationFunction:
    """
    Combine a history of values using a TypeFunction.

    Built-in options:
        - "mean_out_of_typ":
          mean(TypeFunction(x_i)) over all history values x_i

        - "mean_in_typ":
          TypeFunction(mean(x_i)) over all history values x_i

    The history axis is axis=0:
        scalar history: shape (number_of_previous_samples,)
        vector history: shape (number_of_previous_samples, dimension)
    """

    def __init__(self, combination_function, axis=0):
        self.axis = axis

        if isinstance(combination_function, str):
            available_combination_functions = {
                "mean_out_of_typ": self.mean_out_of_typ,
                "mean_in_typ": self.mean_in_typ,
                "sum_out_of_typ": self.sum_out_of_typ,
                "sum_in_typ": self.sum_in_typ,  
            }

            try:
                self.combination = available_combination_functions[
                    combination_function
                ]
            except KeyError as error:
                valid_names = ", ".join(available_combination_functions)
                raise ValueError(
                    f"Unknown combination function {combination_function!r}. "
                    f"Choose one of: {valid_names}."
                ) from error

        elif callable(combination_function):
            # A custom function receives:
            # (values, type_function, *type_parameters, **keyword_parameters)
            self.combination = combination_function

        else:
            raise TypeError(
                "combination_function must be a supported string or a callable."
            )

    def __call__(
        self,
        values,
        type_function,
        *type_parameters,
        **keyword_parameters,
    ):
        """
        Combine values using type_function.

        Parameters after type_function are forwarded to TypeFunction.
        """
        values = np.asarray(values)

        if values.size == 0:
            raise ValueError("Cannot combine an empty history of values.")

        return self.combination(
            values,
            type_function,
            *type_parameters,
            **keyword_parameters,
        )

    def mean_out_of_typ(
        self,
        values,
        type_function,
        *type_parameters,
        **keyword_parameters,
    ):
        """
        Apply the type function first, then compute the mean.

        For example:
            mean_out_of_typ([x1, x2, x3], ReLU)
            = mean(ReLU(x1), ReLU(x2), ReLU(x3))
        """
        transformed_values = type_function(
            values,
            *type_parameters,
            **keyword_parameters,
        )

        # Useful for a constant TypeFunction, which may return one scalar.
        if np.ndim(transformed_values) == 0:
            return transformed_values

        return np.mean(transformed_values, axis=self.axis)

    def mean_in_typ(
        self,
        values,
        type_function,
        *type_parameters,
        **keyword_parameters,
    ):
        """
        Compute the mean first, then apply the type function.

        For example:
            mean_in_typ([x1, x2, x3], ReLU)
            = ReLU(mean(x1, x2, x3))
        """
        mean_value = np.mean(values, axis=self.axis)

        return type_function(
            mean_value,
            *type_parameters,
            **keyword_parameters,
        )

    def sum_out_of_typ(
        self,
        values,
        type_function,
        *type_parameters,
        **keyword_parameters,
    ):
        transformed_values = type_function(
            values,
            *type_parameters,
            **keyword_parameters,
        )
        return np.sum(transformed_values, axis=self.axis)


    def sum_in_typ(
        self,
        values,
        type_function,
        *type_parameters,
        **keyword_parameters,
    ):
        summed_value = np.sum(values, axis=self.axis)
        return type_function(
            summed_value,
            *type_parameters,
            **keyword_parameters,
        )
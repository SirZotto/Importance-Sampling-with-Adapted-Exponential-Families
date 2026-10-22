import numpy as np
import sys
from pathlib import Path
from typing import override
sys.path.append(str(Path(__file__).resolve().parents[1]))


from NaturalParameterGenerator.CombinationFunction import CombinationFunction
from NaturalParameterGenerator.TypeFunction import TypeFunction


class NaturalParameter():
    
    def __init__(self, type_function = "id", combination_function="mean_out_of_typ"):
        
        self.typeFunction = TypeFunction(type_function= type_function)
        self.combinationFunction = CombinationFunction(combination_function= combination_function)
        self.z = self.generate()
        
    def generate(self):
        """Generates the natural parameter with the input provided in the respective __init__
        """
        raise NotImplementedError("Each NaturalParameter subclass must implement generate().")
    
    def __call__(self, x):
        """Calls the respective natural parameter with the input x.
        
        Args:
            x (numpy array/list-like): input for the exponential family/natural parameter
        """
        return self.z(x)
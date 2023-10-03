from .cubic import CubicOracle
from .quadratic import QuadraticOracle
from .ridge import RidgeRegressionOracle
from .multi_logreg import MultiLogRegOracle
from .logreg import LogisticRegressionOracle
from .datacleaning import DataCleaningOracle

__all__ = ['RidgeRegressionOracle', 'LogisticRegressionOracle',
           'DataCleaningOracle', 'MultiLogRegOracle', 'QuadraticOracle',
           'CubicOracle']

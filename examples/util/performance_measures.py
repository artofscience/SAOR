import numpy as np
from abc import abstractmethod, ABC
from array import array

class PerformanceMeasure(ABC):
    """
    Performance measure.
    """
    def __init__(self):
        """On initialisation the name of the measure is set to Default"""
        self.name = "Default PM name"
        self.label = "r'$X$'"
        self.description = "Default PM description"
        self._measure = array('f')

    @property
    def measure(self):
        """I'm the 'measure' property."""
        return self._measure

    @measure.setter
    def measure(self, value):
        self._measure.append(self.set_measure(value))

    @abstractmethod
    def set_measure(self, value):
        ...

    @measure.deleter
    def measure(self):
        del self._measure

class Objective(PerformanceMeasure):
    """
    Objective function value.
    """
    def __init__(self):
        super().__init__()
        self.name = "Objective"
        self.label = r'$f\left[\mathbf{x}^{(k)}\right]$'
        self.description = "Objective function value"

    def set_measure(self, value):
        return value

class Infeasibility(PerformanceMeasure):
    """
    Infeasibility.
    """
    def __init__(self):
        super().__init__()
        self.name = "Infeasibility"
        self.label = r'$\text{max}_{i=1}^{m}\left(0,g_i\left[\mathbf{x}^{(k)}\right]\right)$'
        self.description = "Maximum infeasbility of all constraint functions"

    def set_measure(self, value):
        return np.max([0.0, np.max(value)])

class ObjectiveChange(PerformanceMeasure):
    """
    Absolute change of objective value.
    """
    def __init__(self):
        super().__init__()
        self.name = "Absolute objective change"
        self.label = r'$\left|f\left[\mathbf{x}^{(k)}\right] - f\left[\mathbf{x}^{(k-1)}\right]\right|$'
        self.description = "Absolute objective change"
        self.fold = float('nan')

    def set_measure(self, value):
        a =  np.abs(value - self.fold)
        self.fold = value
        return a

class VariableChangeMax(PerformanceMeasure):
    """
    Maximum absolute variable change.
    """
    def __init__(self):
        super().__init__()
        self.name = "Maximum absolute variable change"
        self.label = r'$\max(\left| \mathbf{x}^{(k)} - \mathbf{x}^{(k-1)}\right|)$'
        self.description = "Maximum absolute variable change"
        self.xold = float('nan')

    def set_measure(self, value):
        a =  np.max(np.abs(value - self.xold))
        self.xold = value
        return a

class VariableChangeNorm(PerformanceMeasure):
    """
    2 Norm of absolute variable change.
    """
    def __init__(self):
        super().__init__()
        self.name = "2 Norm of absolute variable change"
        self.label = r'$\frac{1}{n}\sum_{i}^n \left| x_i^{(k)} - x_i^{(k-1)}\right|$'
        self.description = "2 Norm of absolute variable change"
        self.xold = float('nan')

    def set_measure(self, value):
        a =  np.linalg.norm(np.abs(value - self.xold))
        self.xold = value
        return a

class NonDiscreteness(PerformanceMeasure):
    """
    Measure of non-discreteness aka measure of bounded variables

    Note: currently limited to x_min = 0.0 and x_max = 1.0
    """
    def __init__(self):
        super().__init__()
        self.name = "MND"
        self.label = r'$\frac{4}{n} \sum_{i}^n \tilde{x}_i \left(1-\tilde{x}_i\right)$'
        self.description = "Measure of non-discreteness"

    def set_measure(self, value):
        return 4*np.linalg.norm(value*(1-value))

if __name__ == "__main__":
    f = Objective()
    f.measure = 10
    f.measure = 20

    g = Infeasibility()
    g.measure = [0.4, 0.5, -0.5]
    g.measure = [0.7, 0.3, 2]

    df = ObjectiveChange()
    df.measure = 10
    df.measure = 9.8
    df.measure = 5.1

    mdx = VariableChangeMax()
    ndx = VariableChangeNorm()
    mdx.measure, ndx.measure = np.array([1.0, 3.4, 4.5]), np.array([1.0, 3.4, 4.5])
    mdx.measure, ndx.measure = np.array([1.4, 2.4, 4.5]), np.array([1.4, 2.4, 4.5])
    mdx.measure, ndx.measure = np.array([1.3, 1.4, 3.5]), np.array([1.3, 1.4, 3.5])

    print(f.measure)
    print(g.measure)
    print(df.measure)
    print(mdx.measure)
    print(ndx.measure)
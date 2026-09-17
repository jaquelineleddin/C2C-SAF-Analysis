import numpy as np


class DBELine:
    """
    Creates relation between first and second time with respect to double bound equivalent.
    """

    def __init__(self, dbe_value, limit, point_limit, point_zero, epsilon=0.05):
        """
        Creates function defined by three arguments.

        Parameters
        ----------
        dbe_value : int
            double bound equivalent
        limit : float
            time of second dimension where function align to (Grenzwert)
        point_limit : float
            time of first dimension where lines goes to horizontal limit
        point_zero : float
            time of first dimension where it cuts the x-axis (Nullstelle)
        epsilon : float
            defines the point (point_limit / limit - epsilon) for line to go through
        """
        self.param_b = (np.log(limit) - np.log(epsilon)) / (point_limit - point_zero)
        self.param_a = limit * np.exp(self.param_b * point_zero)
        self.param_c = limit
        self.tolerance = None
        self.dbe = dbe_value

        self.input_str = f'DBE {dbe_value} :({limit}, {point_limit}, {point_zero}, {epsilon})'

    def linefunc(self, x):
        return self.param_c - self.param_a * np.exp(- self.param_b * x)

    def plotLine(self, ax, x, tolerance, label, color, show_tolerance=True):
        self.tolerance = tolerance
        ax.plot(x, self.linefunc(x), color=color, label=label)

        if show_tolerance:
            ax.fill_between(x, self.linefunc(x) - self.tolerance, self.linefunc(x) + self.tolerance, facecolor=color,
                            alpha=0.2)

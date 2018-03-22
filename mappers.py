"""New mappers.
"""
import math

from creamas.rules.mapper import Mapper


class SquareRootDiffMapper(Mapper):
    """Maps feature values by their rooted distance to target given at initialization time.

    The actual computed value is :math:`1 - \sqrt(|t - v|/d)`, where t is target, v is value and
    d is hi - lo.
    """
    def __init__(self, lo, target, hi):
        """
        :param lo: Absolute lowest value for the mapper
        :param target: Target value for the mapper, between [lo, hi]
        :param hi: Absolute highest value for the mapper (must be larger than lo)
        """
        super().__init__()
        self._lo = lo
        self._hi = hi
        self._mid = target
        self._bdiff = hi - lo

    def map(self, value):
        tdiff = 1 - math.sqrt(abs(self._mid - value) / self._bdiff)
        return tdiff if tdiff >= 0.0 else 0.0


class LinearDiffMapper(Mapper):
    """Maps feature values by their linear distance to target given at initialization time.

    The actual computed value is :math:`1 - |t - v|/d`, where t is target, v is value and
    d is hi - lo.
    """
    def __init__(self, lo, target, hi):
        """
        :param lo: Absolute lowest value for the mapper
        :param target: Target value for the mapper, between [lo, hi]
        :param hi: Absolute highest value for the mapper (must be larger than lo)
        """
        super().__init__()
        self._lo = lo
        self._hi = hi
        self._mid = target
        self._bdiff = hi - lo

    def map(self, value):
        tdiff = 1 - (abs(self._mid - value) / self._bdiff)
        return tdiff if tdiff >= 0.0 else 0.0

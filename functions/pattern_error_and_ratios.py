from math import log
from typing import Union, List, Optional


def calculate_harmonic_ratios(X, A, B, C, D):
    """
    Calculate the harmonic ratios based on the XABCD points.

    :param X: Price at point X.
    :param A: Price at point A.
    :param B: Price at point B.
    :param C: Price at point C.
    :param D: Price at point D.
    :return: Tuple of harmonic ratios (ratio_B_XA, ratio_C_B, ratio_D_AB, ratio_D_XA)
    """
    XA = abs(X - A)
    AB = abs(A - B)
    BC = abs(B - C)
    CD = abs(C - D)
    AD = abs(A - D)

    ratio_AB_XA = AB / XA if XA != 0 else None
    ratio_BC_AB = BC / AB if AB != 0 else None
    ratio_CD_BC = CD / BC if BC != 0 else None
    ratio_AD_XA = AD / XA if XA != 0 else None

    return ratio_AB_XA, ratio_BC_AB, ratio_CD_BC, ratio_AD_XA


def calculate_error(actual_ratio: float, pattern_ratio: Optional[Union[float, List[float]]]) -> float:
    if pattern_ratio is None:
        return 0.0

    try:
        log_actual = log(actual_ratio)
    except ValueError:
        return 1e30

    if isinstance(pattern_ratio, list):  # Acceptable range
        log_pat_min = log(pattern_ratio[0])
        log_pat_max = log(pattern_ratio[1])
        assert log_pat_max > log_pat_min

        if log_pat_min <= log_actual <= log_pat_max:
            return 0.0

        error = min(abs(log_actual - log_pat_min), abs(log_actual - log_pat_max))
        return error * 2.0  # Apply multiplier for range-based errors

    elif isinstance(pattern_ratio, float):
        return abs(log_actual - log(pattern_ratio))

    else:
        raise TypeError("Invalid pattern ratio type")

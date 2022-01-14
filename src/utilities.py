import numpy as np

def dbl_colour(days):
    """
    Return a colour corresponding to the number of days to double

    :param days: int
    :return: str
    """
    if days >= 28:
        return "orange"
    elif 0 < days < 28:
        return "red"
    elif days < -28:
        return "green"
    else:
        return "yellow"

def rate_to_dbl(rate):
    return np.log(2) / np.log1p(rate)


def dbl_to_rate(days):
    """
    Return the growth rate required to double / half in days number of days

    :param days: int - if > 0 return rate to double.  If < 0 return rate to half.  If 0, return 1

    :return: float
    """

    if days == np.inf:
        return 0
    else:
        return np.expm1(np.log(2) / days)
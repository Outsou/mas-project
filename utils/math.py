import numpy as np


def combine(num1, num2, num3):
    return [float(num1), float(num2), float(num3)]


def log(a):
    if a <= 0:
        a = 0.000001
    return np.log(a)


def exp(a):
    if a > 100:
        a = 100
    elif a < -100:
        a = -100
    return np.exp(a)


def divide(a, b):
    if b == 0:
        b = 0.000001
    return np.divide(a, b)


def sign(a):
    if a < 0:
        return -1
    elif a > 0:
        return 1
    else:
        return 0


def mdist(a, b):
    return abs(a-b)


def safe_pow(a, b):
    if a == 0 and b < 0:
        return 0
    return pow(a, b)


def abs_sqrt(a):
    return np.sqrt(abs(a))
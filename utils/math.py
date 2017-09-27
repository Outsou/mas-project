import numpy as np
import time

from noise import snoise2, pnoise1, pnoise2

MINVAL = 0.00001


def combine(num1, num2, num3):
    return [float(num1), float(num2), float(num3)]


def safe_exp(x):
    if x < -100:
        x = -100
    elif x > 100:
        x = 100
    return np.exp(x)


def safe_div(a, b):
    if b == 0:
        b = MINVAL
    return np.divide(a, b)


def safe_mod(a, b):
    if b == 0:
        b = MINVAL
    return a % b

def mdist(a, b):
    return np.abs(a-b)


def safe_pow(a, b):
    if a == 0 and b < 0:
        return 0
    return pow(a, b)


def abs_sqrt(a):
    return np.sqrt(abs(a))


def if_then_else(input, output1, output2):
    return output1 if input else output2


def simplex2(x, y):
    return snoise2(x, y)


def perlin1(x):
    return pnoise1(x)


def perlin2(x, y):
    return pnoise2(x, y)



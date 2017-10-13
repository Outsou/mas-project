import math

import numpy as np

from noise import snoise2, pnoise1, pnoise2

MINVAL = 0.00001

HVAL = 10


def rand_eph():
    return np.random.random() * 2 - 1


def _check_hval(x):
    if x > HVAL:
        return HVAL
    elif x < -HVAL:
        return -HVAL
    return x


def combine(num1, num2, num3):
    return [float(num1), float(num2), float(num3)]


def safe_log10(x):
    if x <= 0:
        x = MINVAL
    return math.log10(x)


def safe_log2(x):
    if x <= 0:
        x = MINVAL
    return math.log2(x)


def safe_ln(x):
    if x <= 0:
        x = MINVAL
    return math.log(x)


def safe_exp(x):
    if x < -100:
        x = -100
    elif x > 100:
        x = 100
    return math.exp(x)


def safe_div(a, b):
    if b == 0:
        b = MINVAL
    return a / b


def safe_mod(a, b):
    if b == 0:
        b = MINVAL
    return a % b


def safe_cosh(x):
    x = _check_hval(x)
    return math.cosh(x)


def safe_sinh(x):
    x = _check_hval(x)
    return math.sinh(x)


def mdist(a, b):
    return np.abs(a-b)


def safe_pow(a, b):
    if a == 0 and b < 0:
        return 0
    return pow(a, b)


def abs_sqrt(a):
    return math.sqrt(abs(a))


def if_then_else(input, output1, output2):
    return output1 if input else output2


def simplex2(x, y):
    return snoise2(x, y)


def perlin1(x):
    return pnoise1(x)


def perlin2(x, y):
    return pnoise2(x, y)


def plasma(x, y, t, scale):
    if scale < 0:
        scale = MINVAL
    v1 = math.sin(x * scale + t)
    v2 = math.sin(scale * (x * math.sin(t / 2) + y * math.cos(t / 3)) + t)
    cx = x + 1.0 * math.sin(t / 5)
    cy = y + 1.0 * math.cos(t / 3)
    v3 = math.sin(math.sqrt(scale**2 * (cx**2 + cy**2) + 1) + t)
    return v1 + v2 + v3


def parab(x):
    return 4 * (x - 0.5) ** 2


def avg_sum(x, y):
    return (x + y) / 2

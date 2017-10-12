import math

import numpy as np

from noise import snoise2, pnoise1, pnoise2
from numba import jit, cuda

MINVAL = 0.00001

HVAL = 10


@jit
def _check_hval(x):
    if x > HVAL:
        return HVAL
    elif x < -HVAL:
        return -HVAL
    return x


def combine(num1, num2, num3):
    return [float(num1), float(num2), float(num3)]


@jit
def safe_log10(x):
    if x <= 0:
        x = MINVAL
    return math.log10(float(x))


@jit
def safe_log2(x):
    if x <= 0:
        x = MINVAL
    return math.log(float(x)) / math.log(float(2))


@jit
def safe_exp(x):
    if x < -100:
        x = -100
    elif x > 100:
        x = 100
    return math.exp(float(x))


@jit
def safe_div(a, b):
    if b == 0:
        b = MINVAL
    return a / b


@jit
def safe_mod(a, b):
    if b == 0:
        b = MINVAL
    return a % b


@jit
def safe_cosh(x):
    x = _check_hval(x)
    return math.cosh(x)


@jit
def safe_sinh(x):
    x = _check_hval(x)
    return math.sinh(float(x))


@jit
def mdist(a, b):
    return math.fabs(float(a - b))


@jit(nogil=True)
def safe_pow(a, b):
    if a == 0 and b < 0:
        return 0
    return pow(a, b)


@jit
def abs_sqrt(a):
    return math.sqrt(abs(float(a)))


@jit(nogil=True)
def if_then_else(input, output1, output2):
    return output1 if input else output2


@jit(nogil=True)
def simplex2(x, y):
    return snoise2(x, y)


@jit(nogil=True)
def perlin1(x):
    return pnoise1(x)


@jit(nogil=True)
def perlin2(x, y):
    return pnoise2(x, y)


@jit(nogil=True)
def plasma(x, y, t, scale):
    if scale < 0:
        scale = MINVAL
    v1 = math.sin(x * scale + t)
    v2 = math.sin(scale * (x * math.sin(t / 2) + y * math.cos(t / 3)) + t)
    cx = x + 1.0 * math.sin(t / 5)
    cy = y + 1.0 * math.cos(t / 3)
    v3 = math.sin(math.sqrt(scale**2 * (cx**2 + cy**2) + 1) + t)
    return v1 + v2 + v3


@jit(nogil=True)
def parab(x):
    return 4 * (x - 0.5) ** 2


@jit(nogil=True)
def avg_sum(x, y):
    return (x + y) / 2


def make_f(ind):
    def make_cuda_f(g):
        @cuda.jit
        def f(an_array):
            x, y = cuda.grid(2)
            if x < an_array.shape[0] and y < an_array.shape[1]:
                x_normalized = x / an_array.shape[0] * 2 - 1
                y_normalized = y / an_array.shape[1] * 2 - 1
                an_array[x, y] = g(x_normalized, y_normalized)
        return f

    func = eval('lambda x, y: ' + str(ind))
    jitted_func = jit(func)
    return make_cuda_f(jitted_func)


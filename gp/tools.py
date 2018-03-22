"""Utilities for the :class:`GPImageGenerator` and agents utilizing it.
"""

import gp.primitives as prim
from utils.bitwise import *

from deap import base, tools, gp
import operator
import random
import copy
import math


class Rand(gp.Ephemeral):
    """A helper class to make ephemeral constants in the range [-1, 1].
    """
    ret = float

    @staticmethod
    def func():
        return math.random() * 2 - 1


# Set the attribute to deap's gp-module to circumvent some problems with
# ephemeral constants.
setattr(gp, 'Rand', Rand)


MAX_DEPTH = 8


_primitives = {
    'min': (min, [float, float], float),
    'max': (max, [float, float], float),
    'safe_log2': (prim.safe_log2, [float], float),
    'safe_log10': (prim.safe_log10, [float], float),
    'sin': (math.sin, [float], float),
    'cos': (math.cos, [float], float),
    'safe_sinh': (prim.safe_sinh, [float], float),
    'safe_cosh': (prim.safe_cosh, [float], float),
    'tanh': (math.tanh, [float], float),
    'atan': (math.atan, [float], float),
    'hypot': (math.hypot, [float, float], float),
    'abs': (abs, [float], float),
    'abs_sqrt': (prim.abs_sqrt, [float], float),
    'parab': (prim.parab, [float], float),
    'avg_sum': (prim.avg_sum, [float, float], float),
    'sign': (prim.sign, [float], float),
    'mdist': (prim.mdist, [float, float], float),
    'simplex2': (prim.simplex2, [float, float], float),
    'perlin2': (prim.perlin2, [float, float], float),
    'perlin1': (prim.perlin1, [float], float),
    'plasma': (prim.plasma, [float, float, float, float], float),
    'float_or': (float_or, [float, float], float),
    'float_xor': (float_xor, [float, float], float),
    'float_and': (float_and, [float, float], float)
}


def create_super_pset(bw=True):
    """Create super pset which contains all the primitives for DEAP.

    :param bool bw:
        If ``True`` the returned primitive set is primed to create grey scale images, otherwise it
        will create RGB images.
    :return:
        Created primitive set
    """
    if bw:
        pset = gp.PrimitiveSetTyped("main", [float, float], float)
    else:
        pset = gp.PrimitiveSetTyped("main", [float, float], list)
        pset.addPrimitive(prim.combine, [float, float, float], list)

    # Basic math
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(prim.safe_div, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(prim.safe_mod, [float, float], float)

    # Relational
    pset.addPrimitive(min, [float, float], float)
    pset.addPrimitive(max, [float, float], float)

    # Other math
    pset.addPrimitive(prim.safe_log2, [float], float)
    pset.addPrimitive(prim.safe_log10, [float], float)
    pset.addPrimitive(math.sin, [float], float)
    pset.addPrimitive(math.cos, [float], float)
    pset.addPrimitive(prim.safe_sinh, [float], float)
    pset.addPrimitive(prim.safe_cosh, [float], float)
    pset.addPrimitive(math.tanh, [float], float)
    pset.addPrimitive(math.atan, [float], float)
    pset.addPrimitive(math.hypot, [float, float], float)
    pset.addPrimitive(abs, [float], float)
    pset.addPrimitive(prim.abs_sqrt, [float], float)
    pset.addPrimitive(prim.parab, [float], float)
    pset.addPrimitive(prim.avg_sum, [float, float], float)
    pset.addPrimitive(prim.sign, [float], float)
    pset.addPrimitive(prim.mdist, [float, float], float)

    # Noise
    pset.addPrimitive(prim.simplex2, [float, float], float)
    pset.addPrimitive(prim.perlin2, [float, float], float)
    pset.addPrimitive(prim.perlin1, [float], float)

    # Plasma
    pset.addPrimitive(prim.plasma, [float, float, float, float], float)

    # Bitwise
    pset.addPrimitive(float_or, [float, float], float)
    pset.addPrimitive(float_xor, [float, float], float)
    pset.addPrimitive(float_and, [float, float], float)

    # Constants
    pset.addTerminal(1.6180, float)  # Golden ratio
    pset.addTerminal(math.pi, float)
    pset.addEphemeralConstant('Rand', Rand.func, float)

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    return pset


def create_pset(bw=True):
    """Creates a set of primitives for DEAP.
    """
    return create_super_pset(bw)


def create_sample_pset(bw=True, sample_size=8):
    """Create a sampled pset from all primitives.
    """
    if bw:
        pset = gp.PrimitiveSetTyped("main", [float, float], float)
    else:
        pset = gp.PrimitiveSetTyped("main", [float, float], list)
        pset.addPrimitive(prim.combine, [float, float, float], list)

    # All psets will have basic math and constants

    # Basic math
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(prim.safe_div, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(prim.safe_mod, [float, float], float)

    # Constants
    pset.addTerminal(1.6180, float)  # Golden ratio
    pset.addTerminal(math.pi, float)
    pset.addEphemeralConstant('Rand', Rand.func, float)

    # Other primitives are sampled from the defined primitive set.
    keys = list(_primitives.keys())
    random.shuffle(keys)
    sample_keys = keys[:sample_size]
    for k in sample_keys:
        p = _primitives[k]
        pset.addPrimitive(p[0], p[1], p[2])

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    return pset, sample_keys


def _valid_ind(ind, max_val):
    return ind.height <= max_val


def mutate(individual, pset, expr):
    """Choose a random mutation function from deap. Used with deap's toolbox.
    """
    rand = np.random.rand()
    keep_ind = copy.deepcopy(individual)
    if rand <= 0.25:
        mutated, = gp.mutShrink(individual)
    elif rand <= 0.5:
        mutated, = gp.mutInsert(individual, pset)
    elif rand <= 0.75:
        mutated, = gp.mutNodeReplacement(individual, pset)
    else:
        mutated, = gp.mutUniform(individual, expr, pset)

    if _valid_ind(mutated, MAX_DEPTH):
        return mutated
    else:
        return keep_ind


def subtree_mutate(individual, pset, expr):
    """Choose a random node and generate a subtree to that node using ``expr`` and ``pset``.
    """
    mut_ind = copy.deepcopy(individual)

    while True:
        mutated, = gp.mutUniform(mut_ind, expr, pset)

        if _valid_ind(mutated, MAX_DEPTH):
            return mutated

        mut_ind = copy.deepcopy(individual)


def mate_limit(ind1, ind2):
    keep_inds = [copy.deepcopy(ind) for ind in [ind1, ind2]]
    new_inds = list(gp.cxOnePoint(ind1, ind2))
    for i, ind in enumerate(new_inds):
        if not _valid_ind(ind, MAX_DEPTH):
            new_inds[i] = random.choice(keep_inds)
    return new_inds


def create_toolbox(pset):
    """Creates a DEAP toolbox for genetic programming with DEAP.
    """
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("expr_mut", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("mate", mate_limit)
    toolbox.register("mutate", subtree_mutate, expr=toolbox.expr_mut)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=3,
                     parsimony_size=1.4, fitness_first=True)
    return toolbox

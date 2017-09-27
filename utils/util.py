from utils.serializers import *
from environments import StatEnvironment
from utils.math import *
from utils.bitwise import *
from features import *

from creamas.mp import MultiEnvManager, EnvManager
from creamas.core import Environment
from creamas.util import run
import creamas.features as ft
from creamas.mappers import LinearMapper
from creamas.image import fractal_dimension

from deap import base, tools, gp
import aiomas
import operator
import random
import copy
import math


MAX_DEPTH = 11


def create_environment(num_of_slaves):
    '''Creates a StatEnvironment with slaves.'''
    addr = ('localhost', 5550)

    addrs = []
    for i in range(num_of_slaves):
        addrs.append(('localhost', 5560 + i))

    env_kwargs = {'extra_serializers': [get_type_ser, get_primitive_ser, get_terminal_ser,
                                        get_primitive_set_typed_ser, get_func_ser, get_toolbox_ser,
                                        get_rule_leaf_ser, get_genetic_image_artifact_ser,
                                        get_ndarray_ser, get_dummy_ser], 'codec': aiomas.MsgPack}
    slave_kwargs = [{'extra_serializers': [get_type_ser, get_primitive_ser, get_terminal_ser,
                                           get_primitive_set_typed_ser, get_func_ser, get_toolbox_ser,
                                           get_rule_leaf_ser, get_genetic_image_artifact_ser,
                                           get_ndarray_ser, get_dummy_ser], 'codec': aiomas.MsgPack} for _ in range(len(addrs))]

    menv = StatEnvironment(addr,
                            env_cls=Environment,
                            mgr_cls=MultiEnvManager,
                            logger=None,
                            **env_kwargs)

    ret = run(menv.spawn_slaves(slave_addrs=addrs,
                                slave_env_cls=Environment,
                                slave_mgr_cls=EnvManager,
                                slave_kwargs=slave_kwargs))

    ret = run(menv.wait_slaves(30))
    ret = run(menv.set_host_managers())
    ret = run(menv.is_ready())

    return menv


def get_image_rules(img_shape):
    '''Creates a dictionary of RuleLeafs for images.'''

    rules = {}
    red_rule = RuleLeaf(ft.ImageRednessFeature(), LinearMapper(0, 1, '01'))
    rules['red'] = red_rule
    green_rule = RuleLeaf(ft.ImageGreennessFeature(), LinearMapper(0, 1, '01'))
    rules['green'] = green_rule
    blue_rule = RuleLeaf(ft.ImageBluenessFeature(), LinearMapper(0, 1, '01'))
    rules['blue'] = blue_rule
    complexity_rule = RuleLeaf(ImageComplexityFeature(), LinearMapper(0, 2.3, '01'))
    rules['complexity'] = complexity_rule
    intensity_rule = RuleLeaf(ft.ImageIntensityFeature(), LinearMapper(0, 1, '01'))
    rules['intensity'] = intensity_rule
    benford_rule = RuleLeaf(ImageBenfordsLawFeature(), LinearMapper(0, 1, '01'))
    rules['benford'] = benford_rule
    colorfulness_rule = RuleLeaf(ImageColorfulnessFeature(), LinearMapper(0, 1, '01'))
    rules['colorfulness'] = colorfulness_rule
    entropy_rule = RuleLeaf(ImageEntropyFeature(), LinearMapper(0, 1, '01'))
    rules['entropy'] = entropy_rule
    return rules


def create_pset():
    """Creates a set of primitives for deap.
    """
    pset = gp.PrimitiveSetTyped("main", [float, float], float)
    #pset.addPrimitive(combine, [float, float, float], list)

    # Basic math
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(safe_div, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(safe_mod, [float, float], float)

    # Relational
    pset.addPrimitive(min, [float, float], float)
    pset.addPrimitive(max, [float, float], float)

    # Other math
    #pset.addPrimitive(math.log2, [float], float)
    #pset.addPrimitive(math.log10, [float], float)
    pset.addPrimitive(np.sin, [float], float)
    pset.addPrimitive(np.cos, [float], float)
    pset.addPrimitive(math.sinh, [float], float)
    pset.addPrimitive(math.cosh, [float], float)
    pset.addPrimitive(math.tanh, [float], float)
    pset.addPrimitive(math.atan, [float], float)
    pset.addPrimitive(math.hypot, [float, float], float)
    pset.addPrimitive(np.abs, [float], float)
    pset.addPrimitive(abs_sqrt, [float], float)

    #pset.addPrimitive(exp, [float], float)

    #pset.addPrimitive(safe_pow, [float, float], float)

    #pset.addEphemeralConstant('rand', lambda: np.random.random() * 2 - 1, float)
    pset.addPrimitive(np.sign, [float], float)
    pset.addPrimitive(mdist, [float, float], float)

    # Noise
    pset.addPrimitive(simplex2, [float, float], float)
    pset.addPrimitive(perlin2, [float, float], float)
    pset.addPrimitive(perlin1, [float], float)

    # Bitwise
    pset.addPrimitive(float_or, [float, float], float)
    pset.addPrimitive(float_xor, [float, float], float)
    pset.addPrimitive(float_and, [float, float], float)

    pset.addTerminal(1.6180, float) # Golden ratio
    pset.addTerminal(np.pi, float)

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    return pset


def valid_ind(ind, max_val):
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

    if valid_ind(mutated, MAX_DEPTH):
        return mutated
    else:
        return keep_ind


def mate_limit(ind1, ind2):
    keep_inds = [copy.deepcopy(ind) for ind in [ind1, ind2]]
    new_inds = list(gp.cxOnePoint(ind1, ind2))
    for i, ind in enumerate(new_inds):
        if not valid_ind(ind, MAX_DEPTH):
            new_inds[i] = random.choice(keep_inds)
    return new_inds


def create_toolbox(pset):
    """Creates a deap toolbox for genetic programming with deap.
    """
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=6)
    toolbox.register("mate", mate_limit)
    toolbox.register("mutate", mutate, expr=toolbox.expr)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=3,
                     parsimony_size=1.4, fitness_first=True)
    return toolbox

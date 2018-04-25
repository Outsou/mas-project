from utils.serializers import *
from environments import StatEnvironment
import utils.primitives as m
from utils.bitwise import *
from features import *
from utils.pink_noise import sample_pink

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


class Rand(gp.Ephemeral):

    ret = float

    @staticmethod
    def func():
        return np.random.random() * 2 - 1


# Set the attribute to deap's gp-module to circumvent some problems with
# ephemeral constants.
setattr(gp, 'Rand', Rand)


MAX_DEPTH = 8


primitives = {
    'min': (min, [float, float], float),
    'max': (max, [float, float], float),
    'safe_log2': (m.safe_log2, [float], float),
    'safe_log10': (m.safe_log10, [float], float),
    'sin': (math.sin, [float], float),
    'cos': (math.cos, [float], float),
    'safe_sinh': (m.safe_sinh, [float], float),
    'safe_cosh': (m.safe_cosh, [float], float),
    'tanh': (math.tanh, [float], float),
    'atan': (math.atan, [float], float),
    'hypot': (math.hypot, [float, float], float),
    'abs': (abs, [float], float),
    'abs_sqrt': (m.abs_sqrt, [float], float),
    'parab': (m.parab, [float], float),
    'avg_sum': (m.avg_sum, [float, float], float),
    'sign': (m.sign, [float], float),
    'mdist': (m.mdist, [float, float], float),
    'simplex2': (m.simplex2, [float, float], float),
    'perlin2': (m.perlin2, [float, float], float),
    'perlin1': (m.perlin1, [float], float),
    'plasma': (m.plasma, [float, float, float, float], float),
    'float_or': (float_or, [float, float], float),
    'float_xor': (float_xor, [float, float], float),
    'float_and': (float_and, [float, float], float)
}


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
    """Creates a dictionary of RuleLeafs for images.
    """

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
    fd_aesthetics_rule = RuleLeaf(ImageFDAestheticsFeature(), LinearMapper(0, 1, '01'))
    rules['fd_aesthetics'] = fd_aesthetics_rule
    fd_aesthetics_rule = RuleLeaf(ImageFDAestheticsFeature(), LinearMapper(0, 1, '01'))
    rules['FRD'] = fd_aesthetics_rule
    benford_rule = RuleLeaf(ImageBenfordsLawFeature(), LinearMapper(0, 1, '01'))
    rules['benford'] = benford_rule
    benford_rule = RuleLeaf(ImageBenfordsLawFeature(), LinearMapper(0, 1, '01'))
    rules['BLW'] = benford_rule
    colorfulness_rule = RuleLeaf(ImageColorfulnessFeature(), LinearMapper(0, 1, '01'))
    rules['colorfulness'] = colorfulness_rule
    entropy_rule = RuleLeaf(ImageEntropyFeature(), LinearMapper(0, 1, '01'))
    rules['entropy'] = entropy_rule
    entropy_n_rule = RuleLeaf(ImageEntropyFeature(normalize=True), LinearMapper(0, 1, '01'))
    rules['entropy_n'] = entropy_n_rule
    entropy_n_rule = RuleLeaf(ImageEntropyFeature(normalize=True), LinearMapper(0, 1, '01'))
    rules['ENT'] = entropy_n_rule
    bell_curve_rule = RuleLeaf(ImageBellCurveFeature(), LinearMapper(0, 1, '01'))
    rules['bell_curve'] = bell_curve_rule
    gcf_rule = RuleLeaf(ImageGlobalContrastFactorFeature(), LinearMapper(0, 1, '01'))
    rules['global_contrast_factor'] = gcf_rule
    gcf_rule = RuleLeaf(ImageGlobalContrastFactorFeature(), LinearMapper(0, 1, '01'))
    rules['GCF'] = gcf_rule
    ic_pc_rule = RuleLeaf(ImageMCFeature(), LinearMapper(0, 10, '01'))
    rules['ic_pc'] = ic_pc_rule
    # horizontal symmetry
    hsymm_rule = RuleLeaf(ImageSymmetryFeature(axis=1), LinearMapper(0, 1, '01'))
    rules['hsymm'] = hsymm_rule
    # vertical symmetry
    vsymm_rule = RuleLeaf(ImageSymmetryFeature(axis=2), LinearMapper(0, 1, '01'))
    rules['vsymm'] = vsymm_rule
    # diagonal symmetry
    dsymm_rule = RuleLeaf(ImageSymmetryFeature(axis=4), LinearMapper(0, 1, '01'))
    rules['dsymm'] = dsymm_rule
    # All symmetries
    symm_rule = RuleLeaf(ImageSymmetryFeature(axis=7), LinearMapper(0, 1, '01'))
    rules['symm'] = symm_rule
    symmne_rule = RuleLeaf(ImageSymmetryFeature(axis=7, use_entropy=False), LinearMapper(0, 1, '01'))
    rules['symm_ne'] = symmne_rule
    symmne_rule = RuleLeaf(ImageSymmetryFeature(axis=7, use_entropy=False), LinearMapper(0, 1, '01'))
    rules['SYM'] = symmne_rule
    return rules


def create_super_pset(bw=True):
    """Create super pset which contains all the primitives

    :param bw:
        If ``True`` GP will create grey scale images, otherwise it will create
        RGB images.
    :return:
        Created primitive set
    """
    if bw:
        pset = gp.PrimitiveSetTyped("main", [float, float], float)
    else:
        pset = gp.PrimitiveSetTyped("main", [float, float], list)
        pset.addPrimitive(m.combine, [float, float, float], list)

    # Basic math
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(m.safe_div, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(m.safe_mod, [float, float], float)

    # Relational
    pset.addPrimitive(min, [float, float], float)
    pset.addPrimitive(max, [float, float], float)

    # Other math
    pset.addPrimitive(m.safe_log2, [float], float)
    pset.addPrimitive(m.safe_log10, [float], float)
    pset.addPrimitive(math.sin, [float], float)
    pset.addPrimitive(math.cos, [float], float)
    pset.addPrimitive(m.safe_sinh, [float], float)
    pset.addPrimitive(m.safe_cosh, [float], float)
    pset.addPrimitive(math.tanh, [float], float)
    pset.addPrimitive(math.atan, [float], float)
    pset.addPrimitive(math.hypot, [float, float], float)
    pset.addPrimitive(abs, [float], float)
    pset.addPrimitive(m.abs_sqrt, [float], float)
    pset.addPrimitive(m.parab, [float], float)
    pset.addPrimitive(m.avg_sum, [float, float], float)
    pset.addPrimitive(m.sign, [float], float)
    pset.addPrimitive(m.mdist, [float, float], float)
    #pset.addPrimitive(exp, [float], float)
    #pset.addPrimitive(safe_pow, [float, float], float)

    # Noise
    pset.addPrimitive(m.simplex2, [float, float], float)
    pset.addPrimitive(m.perlin2, [float, float], float)
    pset.addPrimitive(m.perlin1, [float], float)

    # Plasma
    pset.addPrimitive(m.plasma, [float, float, float, float], float)

    # Bitwise
    pset.addPrimitive(float_or, [float, float], float)
    pset.addPrimitive(float_xor, [float, float], float)
    pset.addPrimitive(float_and, [float, float], float)

    # Constants
    #pset.addEphemeralConstant('pink', sample_pink, float)
    pset.addTerminal(1.6180, float)  # Golden ratio
    pset.addTerminal(np.pi, float)
    pset.addEphemeralConstant('Rand', Rand.func, float)
    #pset.addEphemeralConstant('rand', m.rand_eph, float)
    #pset.addEphemeralConstant('rand', rand_eph, float)

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    return pset


def create_pset(bw=True):
    """Creates a set of primitives for deap.
    """
    return create_super_pset(bw)


def create_sample_pset(bw=True, sample_size=8):
    """Create a sampled pset.
    """
    if bw:
        pset = gp.PrimitiveSetTyped("main", [float, float], float)
    else:
        pset = gp.PrimitiveSetTyped("main", [float, float], list)
        pset.addPrimitive(m.combine, [float, float, float], list)

    # All psets will have basic math and constants

    # Basic math
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(m.safe_div, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(m.safe_mod, [float, float], float)

    # Constants
    pset.addTerminal(1.6180, float)  # Golden ratio
    pset.addTerminal(np.pi, float)
    pset.addEphemeralConstant('Rand', Rand.func, float)

    # Other primitives are sampled from the defined primitive set.
    keys = list(primitives.keys())
    random.shuffle(keys)
    sample_keys = keys[:sample_size]
    for k in sample_keys:
        p = primitives[k]
        pset.addPrimitive(p[0], p[1], p[2])

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    return pset, sample_keys


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


def subtree_mutate(individual, pset, expr):
    """Choose a random node and generate a subtree to that node using ``expr``.
    """
    mut_ind = copy.deepcopy(individual)

    while True:
        mutated, = gp.mutUniform(mut_ind, expr, pset)

        if valid_ind(mutated, MAX_DEPTH):
            return mutated

        mut_ind = copy.deepcopy(individual)


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
    toolbox.register("expr_mut", gp.genHalfAndHalf, pset=pset, min_=1, max_=3)
    toolbox.register("mate", mate_limit)
    toolbox.register("mutate", subtree_mutate, expr=toolbox.expr_mut)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=3,
                     parsimony_size=1.4, fitness_first=True)
    return toolbox

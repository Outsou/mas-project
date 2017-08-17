from utils.serializers import *
from environments import StatEnvironment
from utils.math import *
from utils.bitwise import *

from creamas.mp import MultiEnvManager, EnvManager
from creamas.core import Environment
from creamas.util import run
import creamas.features as ft
from creamas.mappers import LinearMapper
from creamas.image import fractal_dimension

from deap import base, tools, gp
import aiomas
import operator


def create_environment(num_of_slaves):
    '''Creates a StatEnvironment with slaves.'''
    addr = ('localhost', 5550)

    addrs = []
    for i in range(num_of_slaves):
        addrs.append(('localhost', 5560 + i))

    env_kwargs = {'extra_serializers': [get_type_ser, get_primitive_ser, get_terminal_ser,
                                        get_primitive_set_typed_ser, get_func_ser, get_toolbox_ser,
                                        get_rule_leaf_ser,
                                        get_ndarray_ser, get_dummy_ser], 'codec': aiomas.MsgPack}
    slave_kwargs = [{'extra_serializers': [get_type_ser, get_primitive_ser, get_terminal_ser,
                                           get_primitive_set_typed_ser, get_func_ser, get_toolbox_ser,
                                           get_rule_leaf_ser,
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
    complexity_rule = RuleLeaf(ft.ImageComplexityFeature(), LinearMapper(0, fractal_dimension(np.ones(img_shape)), '01'))
    rules['complexity'] = complexity_rule
    intensity_rule = RuleLeaf(ft.ImageIntensityFeature(), LinearMapper(0, 1, '01'))
    rules['intensity'] = intensity_rule
    return rules


def create_pset():
    '''Creates a set of primitives for deap.'''
    pset = gp.PrimitiveSetTyped("main", [float, float], list)
    pset.addPrimitive(combine, [float, float, float], list)
    pset.addPrimitive(operator.mul, [float, float], float)
    pset.addPrimitive(divide, [float, float], float)
    pset.addPrimitive(operator.add, [float, float], float)
    pset.addPrimitive(operator.sub, [float, float], float)
    pset.addPrimitive(np.sin, [float], float)
    pset.addPrimitive(np.cos, [float], float)
    #pset.addPrimitive(np.tan, [float], float)
    pset.addPrimitive(min, [float, float], float)
    pset.addPrimitive(max, [float, float], float)
    pset.addPrimitive(np.abs, [float], float)
    #pset.addPrimitive(exp, [float], float)
    #pset.addPrimitive(log, [float], float)
    #pset.addPrimitive(safe_pow, [float, float], float)
    pset.addPrimitive(abs_sqrt, [float], float)
    #pset.addEphemeralConstant('rand', lambda: np.random.random() * 2 - 1, float)
    pset.addPrimitive(sign, [float], float)
    pset.addPrimitive(mdist, [float, float], float)
    pset.addPrimitive(float_or, [float, float], float)
    pset.addPrimitive(float_xor, [float, float], float)
    pset.addPrimitive(float_and, [float, float], float)

    pset.renameArguments(ARG0="x")
    pset.renameArguments(ARG1="y")
    return pset

def mutate(individual, pset, expr):
    '''Choose a random mutation function from deap. Used with deap's toolbox.'''
    rand = np.random.rand()
    if rand <= 0.25:
        return gp.mutShrink(individual),
    elif rand <= 0.5:
        return gp.mutInsert(individual, pset)
    elif rand <= 0.75:
        return gp.mutNodeReplacement(individual, pset)
    return gp.mutUniform(individual, expr, pset)


def create_toolbox(pset):
    '''Creates a deap toolbox for genetic programming with deap.'''
    toolbox = base.Toolbox()
    toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
    toolbox.register("mate", gp.cxOnePoint)
    toolbox.register("mutate", mutate, expr=toolbox.expr)
    toolbox.register("select", tools.selDoubleTournament, fitness_size=3, parsimony_size=1.4, fitness_first=True)
    return toolbox

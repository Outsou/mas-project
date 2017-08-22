from creamas.rules import RuleLeaf
from artifacts import GeneticImageArtifact
from artifacts import DummyArtifact
from deap.gp import Primitive, Terminal, PrimitiveSet, PrimitiveSetTyped
from deap.base import Toolbox

import pickle
from types import FunctionType
from numpy import ndarray


def get_func_ser():
    return FunctionType, pickle.dumps, pickle.loads


def get_primitive_ser():
    return Primitive, pickle.dumps, pickle.loads


def get_terminal_ser():
    return Terminal, pickle.dumps, pickle.loads


def get_primitive_set_ser():
    return PrimitiveSet, pickle.dumps, pickle.loads


def get_primitive_set_typed_ser():
    return PrimitiveSetTyped, pickle.dumps, pickle.loads


def get_toolbox_ser():
    return Toolbox, pickle.dumps, pickle.loads


def get_type_ser():
    return type, pickle.dumps, pickle.loads


def get_rule_leaf_ser():
    return RuleLeaf, pickle.dumps, pickle.loads


def get_genetic_image_artifact_ser():
    return GeneticImageArtifact, pickle.dumps, pickle.loads


def get_ndarray_ser():
    return ndarray, pickle.dumps, pickle.loads


def get_dummy_ser():
    return DummyArtifact, pickle.dumps, pickle.loads

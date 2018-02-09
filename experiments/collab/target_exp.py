"""Collaboration tests with external agent modeling using target information
and dynamic agents.
"""
import pickle
import os
import shutil
import time
import pprint
import logging

import aiomas
import networkx as nx
import numpy as np
import creamas.nx as cnx
from creamas.util import run
from creamas.mp import EnvManager, MultiEnvManager
from creamas import Environment
from creamas.mappers import DoubleLinearMapper
from creamas.rules import RuleLeaf

from utils.serializers import get_serializers
from utils.util import create_super_pset, create_sample_pset, create_toolbox, get_image_rules
from artifacts  import GeneticImageArtifact
from features import *

import experiments.collab.plotting as plott
from experiments.collab.base import CollabEnvironment, CollabSimulation

# Default values for simulation parameters
DEFAULT_PARAMS = {
    'agents': 16,
    'critic_threshold': 0.0,
    'veto_threshold': 0.0,
    'novelty_weight': 0.5,
    'mem_size': 500,
    'search_width': 10,
    'shape': (64, 64),
    'output_shape': (200, 200),
    'model': 'random',  # learning model for choosing collaboration partners
    'pset_sample_size': 8,
    'number_of_steps': 200,
    'population_size': 20,
    'aesthetics': ['entropy', 'complexity'],
    # Bounds for agents' target values for each aesthetic
    'bounds': {'entropy': [0.5, 5.0], 'complexity': [0.5, 2.999]}
}


def _make_rules(rule_names, shape):
    rule_dict = get_image_rules(shape)
    rules = []
    for name in rule_names:
        rules.append(rule_dict[name])
    return rules


def _init_data_folder(data_folder):
    """Initialize the data folder by deleting the existing folder and creating
    new subfolder for averages.
    """
    avgs_folder = os.path.join(data_folder, 'averages')
    shutil.rmtree(data_folder, ignore_errors=True)
    os.makedirs(avgs_folder)
    return avgs_folder


def make_loop_matrix(params):
    """Construct a product of available parameter lists to create a single loop
    which executes every combination of parameter values.

    This function naively assumes that values in params-dictionary that are
    lists are supposed to be executed in different experiments.
    """
    in_loop = []
    for k, v in params.items():
        if type(v) == list:
            in_loop.append((k, v))
    import itertools
    prods = list(itertools.product(*[e[1] for e in in_loop]))
    loop = []
    for p in prods:
        s = {}
        for i in range(len(p)):
            s[in_loop[i][0]] = p[i]
        loop.append(s)
    return loop, len(in_loop)


def _get_default_params(params):
    if params is None:
        params = DEFAULT_PARAMS
    else:
        for key, value in DEFAULT_PARAMS.items():
            if key not in params:
                params[key] = value
    return params


def create_environment(num_of_slaves, save_folder=None):
    """Creates a StatEnvironment with slaves.
    """
    addr = ('localhost', 5550)

    addrs = []
    for i in range(num_of_slaves):
        addrs.append(('localhost', 5560 + i))

    env_kwargs = {'extra_serializers': get_serializers(),
                  'codec': aiomas.MsgPack,
                  'save_folder': save_folder}
    slave_kwargs = [{'extra_serializers': get_serializers(),
                     'codec': aiomas.MsgPack} for _ in range(len(addrs))]

    logger = logging.getLogger("CollaborationEnvironmentLogger")
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    menv = CollabEnvironment(addr,
                             env_cls=Environment,
                             mgr_cls=MultiEnvManager,
                             logger=logger,
                             **env_kwargs)

    ret = run(menv.spawn_slaves(slave_addrs=addrs,
                                slave_env_cls=Environment,
                                slave_mgr_cls=EnvManager,
                                slave_kwargs=slave_kwargs))

    wait_time = 30
    ret = run(menv.wait_slaves(wait_time))
    ret = run(menv.set_host_managers())
    ret = run(menv.is_ready())
    if not ret:
        print("Environment not ready after waiting for it!")

    return menv


def create_agents(agent_cls, menv, params, log_folder, save_folder,
                  pop_size, shape, sample_size):
    ae_list = params['aesthetics']
    super_pset = create_super_pset(bw=True)
    rets = []
    for i in range(params['agents']):
        critic_threshold = params['critic_threshold']
        veto_threshold = params['veto_threshold']
        novelty_weight = params['novelty_weight']
        memsize = params['mem_size']
        search_width = params['search_width']
        shape = params['shape']
        collab_model = params['model'] # Learning model
        output_shape = params['output_shape']
        aesthetic = ae_list[i % len(ae_list)]
        if aesthetic == 'entropy':
            feat = ImageEntropyFeature
        elif aesthetic == 'complexity':
            feat = ImageComplexityFeature
        dlm = DoubleLinearMapper(feat.MIN, aesthetic_target, feat.MAX, '01')
        rules = [RuleLeaf(feat(), dlm)]
        rule_weights = [1.0]
        create_kwargs, funnames = get_create_kwargs(20, shape, 8)
        # print(create_kwargs['pset'])
        ret = aiomas.run(until=menv.spawn(agent_cls,
                                          log_folder=log_folder,
                                          save_folder=save_folder,
                                          artifact_cls=GeneticImageArtifact,
                                          create_kwargs=create_kwargs,
                                          rules=rules,
                                          rule_weights=rule_weights,
                                          memsize=memsize,
                                          critic_threshold=critic_threshold,
                                          veto_threshold=veto_threshold,
                                          novelty_weight=novelty_weight,
                                          search_width=search_width,
                                          output_shape=output_shape,
                                          collab_model=collab_model,
                                          super_pset=super_pset,
                                          aesthetic=aesthetics[0],
                                          novelty_threshold=0.01,
                                          value_threshold=0.01,
                                          pset_names=funnames))
        # print("Created {} with aesthetics: {}".format(ret[1], aesthetics))
        rets.append(ret)

    return rets


def get_create_kwargs(pop_size, shape, sample_size, *args, **kwargs):
    pset, funnames = create_sample_pset(sample_size=sample_size)
    #pset = None
    create_kwargs = {'pset': pset,
                     'toolbox': create_toolbox(pset),
                     'pop_size': pop_size,
                     'shape': shape}
    return create_kwargs, funnames


def create_agent_connections(menv, n_agents):
    """Create fully connected agent society.
    """
    G = nx.Graph()
    G.add_nodes_from(list(range(n_agents)))
    edges = []
    for i in range(0, n_agents - 1):
        for j in range(i + 1, n_agents):
            edges.append((i, j))
    G.add_edges_from(edges)
    cnx.connections_from_graph(menv, G)

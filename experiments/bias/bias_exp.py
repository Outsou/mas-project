"""Simple collaboration tests with external agent modeling.
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

from utils.serializers import get_serializers
from utils.util import create_super_pset, create_sample_pset, create_toolbox, get_image_rules
from artifacts  import GeneticImageArtifact
from features import *

import experiments.collab.plotting as plott
from experiments.bias.base import BiasEnvironment, BiasSimulation

# Default values for simulation parameters
DEFAULT_PARAMS = {
    'agents': 160,
    'critic_threshold': 0.5,
    'veto_threshold': 0.0,
    'novelty_weight': 0.5,
    'mem_size': 500,
    'search_width': 5,
    'shape': (64, 64),
    'output_shape': (200, 200),
    'pset_sample_size': 8,
    'number_of_steps': 200,
    'population_size': 4,
    'aesthetic_list': ['ENT', 'BLW', 'FRD', 'GCF'],
    'bias_prob': 0.2,
    'bias_dist': (0.0, 0.1)
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

    logger = logging.getLogger("BiasEnvironmentLogger")
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)

    menv = BiasEnvironment(addr,
                           env_cls=Environment,
                           mgr_cls=MultiEnvManager,
                           logger=logger,
                           **env_kwargs)

    ret = run(menv.spawn_slaves(slave_addrs=addrs,
                                slave_env_cls=Environment,
                                slave_mgr_cls=EnvManager,
                                slave_kwargs=slave_kwargs))

    ret = run(menv.wait_slaves(30))
    ret = run(menv.set_host_managers())
    ret = run(menv.is_ready())

    return menv


def create_agents(agent_cls, menv, params, log_folder, save_folder,
                  pop_size, shape, sample_size):
    ae_list = params['aesthetic_list']
    super_pset = create_super_pset(bw=True)
    rets = []
    for i in range(params['agents']):
        critic_threshold = params['critic_threshold']
        veto_threshold = params['veto_threshold']
        novelty_weight = params['novelty_weight']
        memsize = params['mem_size']
        search_width = params['search_width']
        shape = params['shape']
        output_shape = params['output_shape']
        aesthetics = [ae_list[i % len(ae_list)]]
        rules = _make_rules(aesthetics, shape)
        rule_weights = [1.0]
        create_kwargs, funnames = get_create_kwargs(pop_size, shape, sample_size)
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
    """Create connected Watts-Strogatz small world graph for *n_agents*.
    """
    # Every agent is connected to 10 nearest neighbors and on average every fifth agent has an extra connection.
    G = nx.generators.newman_watts_strogatz_graph(n=n_agents, k=6, p=0.1)
    cnx.connections_from_graph(menv, G)

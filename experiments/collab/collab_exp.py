"""Simple collaboration tests with external agent modeling.
"""
import pickle
import os
import shutil
import time
import pprint

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
from experiments.collab.base import CollabEnvironment, CollabSimulation

# Default values for simulation parameters
DEFAULT_PARAMS = {
    'agents': 2,
    'critic_threshold': 0.0,
    'veto_threshold': 0.0,
    'novelty_weight': -1,
    'mem_size': 100,
    'search_width': 10,
    'shape': (64, 64),
    'model': 'random',  # learning model for choosing collaboration partners
    'pset_sample_size': 8,
    'aesthetic_list': ['entropy', 'benford', 'fd_aesthetics', 'global_contrast_factor', 'symm']
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


def create_environment(num_of_slaves):
    '''Creates a StatEnvironment with slaves.'''
    addr = ('localhost', 5550)

    addrs = []
    for i in range(num_of_slaves):
        addrs.append(('localhost', 5560 + i))

    env_kwargs = {'extra_serializers': get_serializers(),
                  'codec': aiomas.MsgPack}
    slave_kwargs = [{'extra_serializers': get_serializers(),
                     'codec': aiomas.MsgPack} for _ in range(len(addrs))]

    menv = CollabEnvironment(addr,
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
        collab_model = params['model']
        output_shape = (400, 400)
        aesthetics = [ae_list[i % len(ae_list)]]
        rules = _make_rules(aesthetics, shape)
        rule_weights = [1.0]
        create_kwargs = get_create_kwargs(pop_size, shape, sample_size, i)
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
                                          aesthetic=aesthetics[0]))
        print("Created {} with aesthetics: {}".format(ret[1], aesthetics))
        rets.append(ret)

    return rets


def get_create_kwargs(pop_size, shape, sample_size, i, *args, **kwargs):
    pset = create_sample_pset(i=i, sample_size=sample_size)
    #pset = None
    create_kwargs = {'pset': pset,
                     'toolbox': create_toolbox(pset),
                     'pop_size': pop_size,
                     'shape': shape}
    return create_kwargs


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


def run_experiment(agent_cls, params, num_of_simulations, num_of_steps,
                   sample_size,
                   draw_windows=False, data_folder='collaboration_test_data',
                   report=True):
    """Run experiment.

    :param str data_folder:
        Folder to store results to. Any existing files in the folder will be
        cleared before the run!
    :param bool report:
        Report about intermediate advances in experiment to the stdout.
    """
    start_time = time.time()
    models = ['sgd', 'bandit', 'linear']
    pop_size = 20
    shape = (64, 64)

    # Get default parameter values for keys that are not present in
    # 'params' dictionary, construct replacement parameters for the experiment
    # matrix and save initial parameters.
    params = _get_default_params(params)
    loop, looping_params = make_loop_matrix(params)
    old_params = params.copy()
    log_folder = None
    avgs_folder = _init_data_folder(data_folder)

    run_id = 0
    times = []  # Run times for reporting during the run.

    for replace_params in loop:
        params.update(replace_params)
        run_id += 1
        sim_id = 0
        path = os.path.join(data_folder, str(run_id))
        os.makedirs(path)

        for _ in range(num_of_simulations):
            t1 = time.monotonic()
            sim_id += 1

            # Create multi-environment for the agents
            menv = create_environment(num_of_slaves=4)

            # Create agents to the environment
            super_pset = create_super_pset(bw=True)
            create_agents(agent_cls, menv, params, log_folder, path, pop_size, shape, sample_size, super_pset)

            # Make fully connected graph for agent connections.
            create_agent_connections(menv, params['agents'])

            # Create simulation
            sim = CollabSimulation(menv, log_folder=log_folder)

            if report:
                lr = str(len(str(run_id)))
                ls = str(len(str(num_of_simulations)))
                print(("Initialized simulation setup {:0>"+lr+"}/{:0>"+lr+"} "
                      "run {:0>"+ls+"}/{:0>"+ls+"} with parameters:")
                      .format(run_id, len(loop), sim_id, num_of_simulations))
                pprint.pprint(params, indent=4)

            # RUN SIMULATION
            for i in range(num_of_steps):
                sim.async_step()

            sim.end()

            # Add some printing to the end of the simulation to help
            # gauge time for longer runs.
            if report:
                t2 = time.monotonic()
                total_time = t2 - t1
                times.append(total_time)
                mean_time = np.mean(times)
                runs_left = (len(loop) - run_id) * num_of_simulations +\
                            (num_of_simulations - sim_id)
                est_end_time = time.ctime(time.time() + (mean_time * runs_left))

                print("Run took {:.3f} seconds. Estimated end time at: {}\n"
                      .format(total_time, est_end_time))

        plott.calc_stats_and_plot(path, models, params, num_of_simulations,
                                  replace_params, draw_windows, avgs_folder,
                                  run_id)

    print('Run took {}s'.format(int(np.around(time.time() - start_time))))

    # Some purkka.
    if looping_params == 1:
        key, values = None, None
        for k, v in old_params.items():
            if type(v) == list:
                key, values = k, v
        plott.create_param_graph(avgs_folder, data_folder, key, values,
                                 ['sgd', 'bandit', 'linear'])


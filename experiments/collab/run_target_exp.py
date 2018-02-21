"""Command line script to run collaboration tests.
"""
import logging
import pprint
import random
import copy
import time
import shutil
import socket
import argparse
from argparse import RawTextHelpFormatter
import os
import traceback
import random

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


HOST = socket.gethostname()


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
    'aesthetic_list': ['entropy', 'complexity'],
    # Bounds for agents' target values for each aesthetic
    'bounds': {'entropy': [0.5, 4.5], 'complexity': [0.5, 1.8]},
    'target_adjustment': 'static'
}


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
    LOG_LEVEL = logging.DEBUG
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
        collab_model = params['model'] # Learning model
        output_shape = params['output_shape']
        aesthetic = ae_list[i % len(ae_list)]
        if aesthetic == 'entropy':
            feat = ImageEntropyFeature
        elif aesthetic == 'complexity':
            feat = ImageComplexityFeature
        aesthetic_bounds = params['bounds'][aesthetic]
        aesthetic_target = random.uniform(*aesthetic_bounds)
        dlm = LinearDiffMapper(feat.MIN, aesthetic_target, feat.MAX)
        rules = [RuleLeaf(feat(), dlm)]
        rule_weights = [1.0]
        create_kwargs, funnames = get_create_kwargs(20, shape, 8)
        ret = aiomas.run(until=menv.spawn(agent_cls,
                                          log_folder=log_folder,
                                          save_folder=save_folder,
                                          log_level=LOG_LEVEL,
                                          artifact_cls=GeneticImageArtifact,
                                          aesthetic_target=aesthetic_target,
                                          aesthetic_bounds=aesthetic_bounds,
                                          drifting_speed=params['drift_speed'],
                                          aesthetic_drift_amount=params['drift_amount'],
                                          drifting_prob=params['drift_prob'],
                                          target_adjustment=params['target_adjustment'],
                                          curious_behavior=params['curious_behavior'],
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
                                          aesthetic=aesthetic,
                                          novelty_threshold=0.01,
                                          value_threshold=0.01,
                                          pset_names=funnames))
        rets.append(ret)
        print("Created {:<10} T: {}".format(aesthetic, aesthetic_target))

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


def get_run_id(path):
    path = path if len(path) > 0 else "."
    d = os.listdir(path)
    run_id = 1
    for e in d:
        if os.path.isdir(os.path.join(path, e)):
            run_id += 1
    return run_id


def run_sim(params, save_path, log_folder):
    nslaves = 8
    num_of_steps = params['num_of_steps']
    pop_size = params['population_size']
    shape = params['shape']
    sample_size = params['pset_sample_size']

    with open(os.path.join(save_path, 'rinfo.txt'), 'w') as f:
        f.write("HOST: {}\n\n".format(HOST))
        f.write("PARAMS:\n")
        f.write(pprint.pformat(params))
        f.write("\n\n")

    menv = create_environment(num_of_slaves=nslaves, save_folder=save_path)
    r = create_agents('experiments.collab.base:DriftingGPCollaborationAgent',
                      menv, params, log_folder, save_path, pop_size, shape,
                      sample_size)
    create_agent_connections(menv, params['agents'])

    sim = CollabSimulation(menv,
                           precallback=menv.match_collab_partners,
                           callback=menv.post_cbk,
                           log_folder=log_folder)

    try:
        # RUN SIMULATION
        step_times = []
        for i in range(num_of_steps):
            step_start = time.monotonic()
            sim.async_step()
            step_time = time.monotonic() - step_start
            step_times.append(step_time)
            mean_step_time = np.mean(step_times)
            run_end_time = time.ctime(time.time() +
                                      (mean_step_time * (num_of_steps - (i + 1))))
            print('Step {}/{} finished in {:.3f} seconds. Estimated end time at: {}'
                  .format((i + 1), num_of_steps, step_time, run_end_time))
            with open(os.path.join(save_path, 'rinfo.txt'), 'a') as f:
                f.write('({}) {}: Step {}/{}, estimated end time {}.\n'
                        .format(HOST, time.ctime(time.time()), i + 1, num_of_steps,
                                run_end_time))
        rets = menv.save_artifact_info()
        sim.end()
        with open(os.path.join(save_path, 'rinfo.txt'), 'a') as f:
            f.write('({}) Run finished at {}\n'
                    .format(HOST, time.ctime(time.time())))
    except:
        sim.end()
        # Something bad happened during the run!
        with open('COLLAB_RUN_ERRORS.txt', 'a') as f:
            f.write("HOST: {}\n\n\{}".format(HOST, traceback.format_exc()))
        with open(os.path.join(save_path, 'rinfo.txt'), 'a') as f:
            f.write("\n\n{}\n".format(traceback.format_exc()))
            f.write("({}) RUN CRASHED (Step {}/{})"
                    .format(HOST, i + 1, num_of_steps, time.ctime(time.time())))
        return False
    return True


if __name__ == "__main__":
    # Command line argument parsing
    desc = "Command line script to run collaboration test runs with goal-aware agents."
    parser = argparse.ArgumentParser(description=desc,
                                     formatter_class=RawTextHelpFormatter)
    parser.add_argument('-a', metavar='agents', type=int, dest='agents',
                        help="Number of agents.", default=16)
    parser.add_argument('-s', metavar='steps', type=int, dest='steps',
                        help="Number of simulation steps.", default=200)
    parser.add_argument('-m', metavar='model', type=str, dest='model',
                        default='random',
                        help="Learning model to be used.\n"
                             "random: no learning, collaboration are chosen randomly\n"
                             "Q0: Gets reward 1 if both agents in collaboration passed\n"
                             "the artifact, 0 otherwise. How often collaboration succeeds?\n"
                             "simple-Q: Reward is the evaluation the agent gives to\n"
                             "the artifact created in collaboration. How much do I gain\n"
                             "from collaboration personally?\n"
                             "hedonic-Q: Reward is own evaluation of artifact created by\n"
                             "another agent. Learns only from artifacts created by a single\n"
                             "agent. Who creates artifacts that are interesting to me?\n"
                             "altruistic-Q: Reward is own artifact's evaluation by another\n"
                             "agent. Who likes my artifacts?\n"
                             "lr: Trains a linear regression model for each neighbour based\n"
                             "on both evaluations of own artifacts by the neighbour and\n"
                             "the neighbour's evaluations of its own artifacts. Who would like\n"
                             "the artifacts in this initial population I have created?\n"
                             "state-Q: Reward is own evaluation of artifact created by another\n"
                             "agent. Evaluation is done simultaneously with multiple different\n"
                             "targets, which are mapped to states for Q-learning. How much would\n"
                             "I like other agents' artifacts, if I had a different target for\n"
                             "my evaluation function?")
    parser.add_argument('-n', metavar='novelty', type=float, dest='novelty',
                        help="Novelty weight.", default=0.5)
    parser.add_argument('-q', metavar='drift amount', type=float,
                        dest='drift_amount', help="Drifting amount, proportional to each agent's aesthetic value's bounds.",
                        default=0.2)
    parser.add_argument('-p', metavar='drift probability', type=float,
                        dest='drift_prob',
                        help="Drifting probability, how likely agents choose a new aesthetic target on each simulation step.",
                        default=0.0)
    parser.add_argument('-t', metavar='drift speed', type=int,
                        dest='drift_speed',
                        help="Drifting speed, how many steps it takes for agents to reach their new aesthetic target.",
                        default=1)
    parser.add_argument('-g', metavar='target adjustment', type=str,
                        dest='target_adjustment',
                        help="How agents adjust their collaboration goals\n"
                             "static: no adjustment\n"
                             "selector: selector adjusts its target to what it perceives best w.r.t. selected agent\n"
                             "selected: selected adjusts its target to what it perceives best w.r.t. selector agent",
                        default='static')
    parser.add_argument('-c', metavar='curious behavior', type=str,
                        dest='curious_behavior',
                        help="Agent's curious behavior, only applicable with state-Q\n"
                             "static: No curiosity\n"
                             "personal: Only personal artifacts accumulate curiosity\n"
                             "social: All seen artifacts (personal or others) accumulate curiosity",
                        default='static')
    parser.add_argument('-l', metavar='folder', type=str, dest='save_folder',
                        help="Base folder to save the test run. Actual save "
                             "folder is created as a subfolder to the base " 
                             "folder.",
                        default="runs")
    parser.add_argument('-r', metavar='run ID', type=int, dest='run_id',
                        help="Run ID, if needed to set manually.", required=False)
    parser.add_argument('-d', metavar='number of runs', type=int, dest='n_runs',
                        help="Number of individual runs to be done", default=1)

    args = parser.parse_args()

    # DEFINE TEST PARAMETERS
    params = DEFAULT_PARAMS
    params['agents'] = args.agents
    params['novelty_weight'] = args.novelty
    params['num_of_steps'] = args.steps
    params['model'] = args.model
    params['drift_amount'] = args.drift_amount
    params['drift_prob'] = args.drift_prob
    params['drift_speed'] = args.drift_speed
    params['target_adjustment'] = args.target_adjustment
    params['curious_behavior'] = args.curious_behavior
    base_path = os.path.join(".", args.save_folder, args.model)
    os.makedirs(base_path, exist_ok=True)
    log_folder = 'foo'
    number_of_runs = args.n_runs
    finished_runs = 0
    try_runs = 0
    print("{} preparing for {} run(s).".format(HOST, number_of_runs))

    # RUN SIMULATION
    while finished_runs < number_of_runs and try_runs < number_of_runs * 2:
        try_runs += 1
        run_id = args.run_id if args.run_id is not None else get_run_id(base_path)

        # CREATE SIMULATION AND RUN
        run_folder = 'r{:0>4}m{}a{}e{}i{}'.format(
            run_id, args.model, params['agents'], len(params['aesthetic_list']),
            params['num_of_steps'])
        if len(base_path) > 0:
            run_folder = os.path.join(base_path, run_folder)
            log_folder = run_folder

        # Error if the run folder exists for some reason. Should not happen if
        # no additional runs are spawned in the same folder.
        os.makedirs(run_folder, exist_ok=False)
        print("Initializing run with {} agents, {} aesthetic measures, {} model, "
              "{} steps.".format(args.agents, len(params['aesthetic_list']),
                                 args.model, args.steps))
        print("Saving run output to {}".format(run_folder))
        success = run_sim(params, run_folder, log_folder)
        finished_runs += success

    print("All runs finished. Exiting!")

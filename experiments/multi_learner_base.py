'''Tests a single agent modeling multiple connected agents.
Test is run simultaneously for several different learning methods.'''

from utils.util import create_environment
from artifacts  import DummyArtifact
from features import DummyFeature
from utils.stats import create_graphs, calculate_averages, create_param_graph

from creamas.rules.rule import RuleLeaf
from creamas.mappers import GaussianMapper
import creamas.nx as cnx
from creamas.core.simulation import Simulation

import aiomas
import networkx as nx
import numpy as np
import pickle
import os
import shutil
import time


def create_img_name(replace_params):
    img_name = ""
    for k, v in replace_params.items():
        img_name += "{}={}".format(k, v)
    img_name += ".png"
    return img_name


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


def run_experiment(params, num_of_simulations, num_of_steps,
                   draw_windows=False, data_folder='multi_test_data',
                   report=True):
    """Run experiment.

    :param str data_folder:
        Folder to store results to. Any existing files in the folder will be
        cleared before the run!
    :param bool report:
        Report about intermediate advances in experiment to the stdout.
    """
    import pprint

    # Default values
    defaults = {
        'agents': 6,
        'features': 5,
        'common_features': 5,  # How many features the creator agent observes
        'std': 0.2,  # Standard deviation for preferences
        'search_width': 10,
        'change_speed': 0,  # How fast preferences change continuously
        'instant_steps': 500000,  # How often instant preference change happens
        'instant_amount': 0,  # Amount of change in instant change
        'reg_weight': 0, # How much weight is given to regularization
        'novelty_weight': 0, # How much weight is given to novelty in evaluation
        'memsize': 0,
        'send_prob': 0 # Probability that non-active agent creates and sends an artifact
    }

    models = ['sgd', 'bandit', 'linear']

    if params is None:
        params = defaults
    else:
        for key, value in defaults.items():
            if key not in params:
                params[key] = value

    start_time = time.time()

    # Environment and simulation
    # log_folder = 'multi_test_logs'
    log_folder = None
    #shutil.rmtree(log_folder, ignore_errors=True)
    avgs_folder = _init_data_folder(data_folder)

    run_id = 0
    times = []

    # Construct replacement parameters for the experiment matrix and save
    # initial parameters.
    loop, looping_params = make_loop_matrix(params)
    old_params = params.copy()

    for replace_params in loop:
        params.update(replace_params)
        create_kwargs = {'length': params['features']}
        run_id += 1
        path = os.path.join(data_folder, str(run_id))
        os.makedirs(path)
        sim_id = 0

        for _ in range(num_of_simulations):
            t1 = time.monotonic()
            sim_id += 1
            menv = create_environment(num_of_slaves=4)

            active = True

            for _ in range(params['agents']):
                rules = []

                for i in range(params['features']):
                    rules.append(RuleLeaf(DummyFeature(i), GaussianMapper(np.random.rand(), params['std'])))
                # rules.append(RuleLeaf(DummyFeature(0), GaussianMapper(0.4, std)))
                # rules.append(RuleLeaf(DummyFeature(1), GaussianMapper(0.8, std)))
                # rules.append(RuleLeaf(DummyFeature(2), GaussianMapper(0.1, std)))

                # rule_weights = [0.1, 0.3, 0.6]
                rule_weights = []
                for _ in range(len(rules)):
                    rule_weights.append(np.random.random())

                if active:
                    # active agent only uses common_features number of the rules
                    ret = aiomas.run(until=menv.spawn('agents:MultiAgent',
                                                      log_folder=log_folder,
                                                      data_folder=path,
                                                      artifact_cls=DummyArtifact,
                                                      create_kwargs=create_kwargs,
                                                      rules=rules[:params['common_features']],
                                                      rule_weights=rule_weights[:params['common_features']],
                                                      std=params['std'],
                                                      active=active,
                                                      search_width=params['search_width'],
                                                      reg_weight=params['reg_weight'],
                                                      novelty_weight=params['novelty_weight'],
                                                      memsize=params['memsize']))
                else:
                    # Generate a rule vec with negative and positive change_speed elements
                    rule_vec = np.random.choice([-params['change_speed'], params['change_speed']], params['features'])
                    ret = aiomas.run(until=menv.spawn('agents:MultiAgent',
                                                      log_folder=log_folder,
                                                      data_folder=path,
                                                      artifact_cls=DummyArtifact,
                                                      create_kwargs=create_kwargs,
                                                      rules=rules,
                                                      rule_weights=rule_weights,
                                                      std=params['std'],
                                                      active=active,
                                                      search_width=params['search_width'],
                                                      rule_vec=rule_vec,
                                                      novelty_weight=params['novelty_weight'],
                                                      memsize=params['memsize'],
                                                      send_prob=params['send_prob']
                                                      ))
                active = False

            # Connect everyone to the main agent
            G = nx.Graph()
            G.add_nodes_from(list(range(params['agents'])))
            edges = [(0, x) for x in range(1, params['agents'])]
            G.add_edges_from(edges)

            cnx.connections_from_graph(menv, G)

            sim = Simulation(menv, log_folder=log_folder)

            if report:
                lr = str(len(str(run_id)))
                ls = str(len(str(num_of_simulations)))
                print(("Initialized simulation setup {:0>"+lr+"}/{:0>"+lr+"} "
                      "run {:0>"+ls+"}/{:0>"+ls+"} with parameters:")
                      .format(run_id, len(loop), sim_id, num_of_simulations))
                pprint.pprint(params, indent=4)

            # RUN SIMULATION
            for i in range(1, num_of_steps + 1):
                if i % params['instant_steps'] == 0:
                    menv.cause_change(params['instant_amount'])
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

                print(("Run took {:.3f} seconds. Estimated end time at: {}\n")
                      .format(total_time, est_end_time))

        avg_stats = calculate_averages(path, models)

        title = 'Connections: {}, features: {}, search width: {}, runs: {}' \
            .format(params['agents'] - 1, params['features'], params['search_width'], num_of_simulations)

        img_name = create_img_name(replace_params)
        create_graphs(path, 10, title, img_name, avg_stats, models, draw_windows)
        pickle.dump(avg_stats, open(os.path.join(avgs_folder, 'avgs{}.p'.format(run_id)), 'wb'))

    print('Run took {}s'.format(int(np.around(time.time() - start_time))))

    # Some purkka.
    if looping_params == 1:
        key, values = None, None
        for k, v in old_params.items():
            if type(v) == list:
                key, values = k, v
        create_param_graph(avgs_folder, data_folder, key, values, ['sgd', 'bandit', 'linear'])


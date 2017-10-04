"""Tests all agents modeling multiple connected agents. An agent only has one model."""

from utils.util import create_environment
from artifacts  import DummyArtifact
from features import DummyFeature
from experiments.multi_learner_base import  make_loop_matrix
from utils.stats import calculate_setup_averages, reward_graph

from creamas.rules.rule import RuleLeaf
from creamas.mappers import GaussianMapper
import creamas.nx as cnx
from creamas.core.simulation import Simulation

import aiomas
import networkx as nx
import numpy as np
import os
import time
import shutil


def run_experiment(params, num_of_simulations, num_of_steps,
                   draw_windows=False, data_folder='single_test_data',
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
        'agent_features': 5,  # How many features the creator agent observes
        'std': 0.2,  # Standard deviation for preferences
        'search_width': 10,
        'change_speed': 0,  # How fast preferences change continuously
        'instant_steps': 500000,  # How often instant preference change happens
        'instant_amount': 0,  # Amount of change in instant change
        'reg_weight': 0, # How much weight is given to regularization
        'novelty_weight': 0, # How much weight is given to novelty in evaluation
        'memsize': 0,
        'send_prob': 0, # Probability that non-active agent creates and sends an artifact
        'model': 'Q',
        'learn_from_received': False
    }

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
    shutil.rmtree(data_folder, ignore_errors=True)

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
        path = os.path.join(data_folder, 'setup' + str(run_id))
        os.makedirs(path)
        sim_id = 0

        # Save setup parameter information
        with open(path + '.txt', 'w') as fout:
            pprint.pprint(params, fout, indent=4)

        for _ in range(num_of_simulations):
            t1 = time.monotonic()
            sim_id += 1
            menv = create_environment(num_of_slaves=4)

            for _ in range(params['agents']):
                rules = []
                # Randomly select rules for this agent
                rule_idx = np.random.choice(range(params['features']), params['agent_features'], replace=False)
                for i in rule_idx:
                    rules.append(RuleLeaf(DummyFeature(i), GaussianMapper(np.random.rand(), params['std'])))

                rule_weights = []
                for _ in range(len(rules)):
                    rule_weights.append(np.random.random())

                ret = aiomas.run(until=menv.spawn('agents:SingleAgent',
                                                  log_folder=log_folder,
                                                  data_folder=path,
                                                  artifact_cls=DummyArtifact,
                                                  create_kwargs=create_kwargs,
                                                  rules=rules,
                                                  rule_weights=rule_weights,
                                                  std=params['std'],
                                                  active=True,
                                                  search_width=params['search_width'],
                                                  reg_weight=params['reg_weight'],
                                                  novelty_weight=params['novelty_weight'],
                                                  memsize=params['memsize'],
                                                  own_folder=True,
                                                  learn_from_received=params['learn_from_received'],
                                                  model=params['model']))

            # Create a fully connected graph
            G = nx.complete_graph(params['agents'])
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
                menv.finalize_step()

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

    setup_avg_stats = calculate_setup_averages(data_folder)

    print('Run took {}s'.format(int(np.around(time.time() - start_time))))

    for i in range(len(setup_avg_stats)):
        reward_graph(setup_avg_stats[i], 10, os.path.join(data_folder, 'setup' + str(i+1)))

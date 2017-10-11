'''Tests a single agent modeling multiple connected agents.
Test is run simultaneously for several different learning methods.'''

from artifacts import GeneticImageArtifact
from utils.util import create_toolbox, create_pset, create_environment, get_image_rules
from utils.stats import create_graphs, calculate_averages, create_param_graph
from experiments.multi_learner_base import create_img_name, _init_data_folder, make_loop_matrix

import creamas.nx as cnx
from creamas.core.simulation import Simulation

import aiomas
import networkx as nx
import numpy as np
import pickle
import os
import time


def run_experiment(params, num_of_simulations, num_of_steps,
                   draw_windows=False, data_folder='multi_gp_image_data',
                   report=True):
    """Run experiment.

    :param str data_folder:
        Folder to store results to. Any existing files in the folder will be
        cleared before the run!
    :param bool report:
        Report about intermediate advances in experiment to the stdout.
    """
    import pprint

    pset = create_pset()
    create_kwargs = {'pset': pset,
                     'toolbox': create_toolbox(pset),
                     'pop_size': 20,
                     'shape': (64, 64)}

    # Default values
    defaults = {
        'main_feature': 'global_contrast_factor',
        'common_features': 5,  # How many features the creator agent observes
        'search_width': 10,
        'change_speed': 0,  # How fast preferences change continuously
        'instant_steps': 500000,  # How often instant preference change happens
        'instant_amount': 0,  # Amount of change in instant change
        'reg_weight': 0, # How much weight is given to regularization
        'novelty_weight': 0, # How much weight is given to novelty in evaluation
        'memsize': 0,
        'send_prob': 0, # Probability that non-active agent creates and sends an artifact
        'rule_names': ('global_contrast_factor', 'benford', 'entropy', 'fd_aesthetics'),
        'create_kwargs': create_kwargs
    }

    models = ['bandit', 'linear']

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
        run_id += 1
        path = os.path.join(data_folder, str(run_id))
        os.makedirs(path)
        sim_id = 0
        rule_dict = get_image_rules(params['create_kwargs']['shape'])

        for _ in range(num_of_simulations):
            t1 = time.monotonic()
            sim_id += 1
            menv = create_environment(num_of_slaves=4)

            # Create main agent
            ret = aiomas.run(until=menv.spawn('agents:MultiAgent',
                                              log_folder=log_folder,
                                              data_folder=path,
                                              artifact_cls=GeneticImageArtifact,
                                              create_kwargs=params['create_kwargs'],
                                              rules=[rule_dict[params['main_feature']]],
                                              active=True,
                                              search_width=params['search_width'],
                                              reg_weight=params['reg_weight'],
                                              novelty_weight=params['novelty_weight'],
                                              memsize=params['memsize']))

            # Create other agents
            for rule_name in params['rule_names']:
                ret = aiomas.run(until=menv.spawn('agents:MultiAgent',
                                                  log_folder=log_folder,
                                                  data_folder=path,
                                                  artifact_cls=GeneticImageArtifact,
                                                  create_kwargs=params['create_kwargs'],
                                                  rules=[rule_dict[rule_name]],
                                                  active=False,
                                                  search_width=params['search_width'],
                                                  reg_weight=params['reg_weight'],
                                                  novelty_weight=params['novelty_weight'],
                                                  memsize=params['memsize']))

            # Connect everyone to the main agent
            G = nx.Graph()
            num_of_agents = len(params['rule_names']) + 1
            G.add_nodes_from(list(range(num_of_agents)))
            edges = [(0, x) for x in range(1, num_of_agents)]
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


            step_times = []
            # RUN SIMULATION
            for i in range(1, num_of_steps + 1):
                step_start = time.monotonic()
                if i % params['instant_steps'] == 0:
                    menv.cause_change(params['instant_amount'])
                sim.async_step()
                step_time = time.monotonic() - step_start
                step_times.append(step_time)
                mean_step_time = np.mean(step_times)
                setup_end_time = time.ctime(time.time() + (mean_step_time * (num_of_steps - i)))
                print('Step {}/{} finished in {:.3f} seconds. Estimated setup end time at: {}'
                      .format(i, num_of_steps, step_time, setup_end_time))

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
            .format(num_of_agents - 1, len(params['rule_names']), params['search_width'], num_of_simulations)

        img_name = 'setup' + str(run_id)
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


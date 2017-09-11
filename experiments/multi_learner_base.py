'''Tests a single agent modeling multiple connected agents.
Test is run simultaneously for several different learning methods.'''

from utils.util import create_environment
from artifacts  import DummyArtifact
from features import DummyFeature

from creamas.rules.rule import RuleLeaf
from creamas.mappers import GaussianMapper
import creamas.nx as cnx
from creamas.core.simulation import Simulation
from creamas.util import run

import aiomas
import networkx as nx
import numpy as np
import pickle
import os
import matplotlib.pyplot as plt
import shutil
import time


def calculate_averages(folder):
    '''Calculates average stats from stat files in folder.'''
    keys_to_avg = [('sgd', 'rewards'), ('sgd', 'chose_best'),
                   ('bandit', 'rewards'), ('bandit', 'chose_best'),
                   ('linear', 'rewards'), ('linear', 'chose_best'),
                   ('poly', 'rewards'), ('poly', 'chose_best'),
                   'random_rewards', 'max_rewards']

    files = os.listdir(folder)
    first_stats = pickle.load(open(os.path.join(folder, files[0]), 'rb'))
    avg_mul = 1 / len(files)
    avg_stats = {}

    for key in keys_to_avg:
        if type(key) is tuple:
            if key[0] not in avg_stats:
                avg_stats[key[0]] = {}
            avg_stats[key[0]][key[1]] = np.array(first_stats[key[0]][key[1]]) * avg_mul
        else:
            avg_stats[key] = np.array(first_stats[key]) * avg_mul

    for i in range(1, len(files)):
        stats = pickle.load(open(os.path.join(folder, files[i]), 'rb'))

        for key in keys_to_avg:
            if type(key) is tuple:
                avg_stats[key[0]][key[1]] = avg_stats[key[0]][key[1]] + np.array(stats[key[0]][key[1]]) * avg_mul
            else:
                avg_stats[key] = avg_stats[key] + np.array(stats[key]) * avg_mul

    return avg_stats


def create_graphs(folder, window_size, title, file_name, stats, draw_windows=False):
    def create_graph(models, maximums, ylabel, title, random=None):
        x = []
        last_idx = len(models[0][1]) - 1

        random_sums = []

        model_sums = {}
        for model in models:
            model_sums[model[0]] = []

        i = 0
        while i < last_idx:
            # Don't use non-complete window
            if i + window_size - 1 > last_idx:
                break

            if draw_windows:
                window_start = i
            else:
                window_start = 0

            maximum = np.sum(maximums[window_start:i + window_size])

            for model in models:
                model_sums[model[0]].append(np.sum(model[1][window_start:i + window_size]) / maximum)

            if random is not None:
                random_reward = np.sum(random[window_start:i + window_size])
                random_sums.append(random_reward / maximum)

            i += window_size

            x.append(i)

        # Draw the graph
        if random is not None:
            plt.plot(x, random_sums, label='Random')

        for model in models:
            plt.plot(x, model_sums[model[0]], label=model[0])

        plt.ylabel(ylabel)
        plt.legend()
        plt.title(title)
        # plt.show()
        path = os.path.split(folder)[0]
        plt.savefig(os.path.join(path, file_name))
        plt.close()

    # Create reward graph
    models = [('SGD', stats['sgd']['rewards']),
              ('Q', stats['bandit']['rewards']),
              ('linear', stats['linear']['rewards'])]
              #('poly', stats['poly']['rewards'])]

    create_graph(models,
                 stats['max_rewards'],
                 'Reward percentage',
                 title,
                 stats['random_rewards'])

    # Create optimal choices graph
    # create_graph(avg_stats['sgd']['chose_best'],
    #              avg_stats['bandit']['chose_best'],
    #              avg_stats['linear']['chose_best'],
    #              np.ones(len(avg_stats['sgd']['chose_best'])),
    #              'Optimal choices',
    #              title)

def create_param_graph(avgs_folder, save_folder, param_name, param_vals, models):
    files = list(sorted(os.listdir(avgs_folder)))
    random_rewards = []

    rewards = {}
    for model in models:
        rewards[model] = []

    for file in files:
        stats = pickle.load(open(os.path.join(avgs_folder, file), 'rb'))
        max_reward = np.sum(stats['max_rewards'])
        for model in models:
            rewards[model].append(np.sum(stats[model]['rewards']) / max_reward)
        random_rewards.append(np.sum(stats['random_rewards']) / max_reward)

    plt.plot(param_vals, random_rewards, label='random')

    for model in models:
        plt.plot(param_vals, rewards[model], label=model)

    plt.legend()
    plt.xlabel(param_name)
    plt.ylabel('Reward %')
    plt.savefig(os.path.join(save_folder, '{}.png'.format(param_name)))
    plt.close()


def run_experiment(params, loop, num_of_simulations, num_of_steps, draw_windows=False):

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
        'reg_weight': 0
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

    data_folder = 'multi_test_data'
    avgs_folder = os.path.join(data_folder, 'averages')
    shutil.rmtree(data_folder, ignore_errors=True)
    os.makedirs(avgs_folder)
    run_id = 0

    for val in loop[1]:
        params[loop[0]] = val
        create_kwargs = {'length': params['features']}
        run_id += 1
        path = os.path.join(data_folder, str(run_id))
        os.makedirs(path)

        for _ in range(num_of_simulations):

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
                                                      reg_weight=params['reg_weight']))
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
                                                      rule_vec=rule_vec))
                print(ret)
                active = False

            # Connect everyone to the main agent
            G = nx.Graph()
            G.add_nodes_from(list(range(params['agents'])))
            edges = [(0, x) for x in range(1, params['agents'])]
            G.add_edges_from(edges)

            cnx.connections_from_graph(menv, G)

            sim = Simulation(menv, log_folder=log_folder)

            for i in range(num_of_steps):
                if i % params['instant_steps'] == 0:
                    menv.cause_change(params['instant_amount'])
                sim.async_step()


            sim.end()

        avg_stats = calculate_averages(path)
        title = 'Connections: {}, features: {}, search width: {}, runs: {}' \
            .format(params['agents'] - 1, params['features'], params['search_width'], num_of_simulations)
        create_graphs(path, 10, title, '{}_{}.png'.format(loop[0], val), avg_stats, draw_windows)
        pickle.dump(avg_stats, open(os.path.join(avgs_folder, 'avgs{}.p'.format(run_id)), 'wb'))

    print('Run took {}s'.format(int(np.around(time.time() - start_time))))
    create_param_graph(avgs_folder, data_folder, loop[0], loop[1], ['sgd', 'bandit', 'linear'])
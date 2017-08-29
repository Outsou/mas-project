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


def create_graphs(folder, window_size, title):
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

            maximum = np.sum(maximums[0:i + window_size])

            for model in models:
                model_sums[model[0]].append(np.sum(model[1][0:i + window_size]) / maximum)

            if random is not None:
                random_reward = np.sum(random[0:i + window_size])
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
        plt.show()
        plt.close()

    avg_stats = calculate_averages(folder)

    # Create reward graph
    models = [('SGD', avg_stats['sgd']['rewards']),
              ('Q', avg_stats['bandit']['rewards']),
              ('linear', avg_stats['linear']['rewards']),
              ('poly', avg_stats['poly']['rewards'])]

    create_graph(models,
                 avg_stats['max_rewards'],
                 'Reward percentage',
                 title,
                 avg_stats['random_rewards'])

    # Create optimal choices graph
    # create_graph(avg_stats['sgd']['chose_best'],
    #              avg_stats['bandit']['chose_best'],
    #              avg_stats['linear']['chose_best'],
    #              np.ones(len(avg_stats['sgd']['chose_best'])),
    #              'Optimal choices',
    #              title)


if __name__ == "__main__":
    start_time = time.time()

    # Parameters
    num_of_agents = 6
    num_of_features = 3
    std = 0.2
    search_width = 10

    num_of_simulations = 40
    num_of_steps = 1000

    create_kwargs = {'length': num_of_features}

    # Environment and simulation

    log_folder = 'multi_test_logs'
    shutil.rmtree(log_folder, ignore_errors=True)

    for _ in range(num_of_simulations):

        menv = create_environment(num_of_slaves=4)

        active = True

        for _ in range(num_of_agents):
            rules = []

            for i in range(num_of_features):
                rules.append(RuleLeaf(DummyFeature(i), GaussianMapper(np.random.rand(), std)))
            # rules.append(RuleLeaf(DummyFeature(0), GaussianMapper(0.4, std)))
            # rules.append(RuleLeaf(DummyFeature(1), GaussianMapper(0.8, std)))
            # rules.append(RuleLeaf(DummyFeature(2), GaussianMapper(0.1, std)))

            # rule_weights = [0.1, 0.3, 0.6]
            rule_weights = []
            for _ in range(len(rules)):
                rule_weights.append(np.random.random())

            ret = aiomas.run(until=menv.spawn('agents:MultiAgent',
                                              log_folder=log_folder,
                                              artifact_cls=DummyArtifact,
                                              create_kwargs=create_kwargs,
                                              rules=rules,
                                              rule_weights=rule_weights,
                                              std=std,
                                              active=active,
                                              search_width=search_width))
            print(ret)
            if active:
                active_folder = run(ret[0].get_log_folder())
            active = False

        # Connect everyone to the main agent
        G = nx.Graph()
        G.add_nodes_from(list(range(num_of_agents)))
        edges = [(0, x) for x in range(1, num_of_agents)]
        G.add_edges_from(edges)

        cnx.connections_from_graph(menv, G)

        sim = Simulation(menv, log_folder=log_folder)
        sim.async_steps(num_of_steps)
        sim.end()

    print('Run took {}s'.format(int(np.around(time.time() - start_time))))
    title = 'Connections: {}, features: {}, search width: {}, runs: {}'\
        .format(num_of_agents - 1, num_of_features, search_width, num_of_simulations)
    create_graphs(active_folder, 10, title)

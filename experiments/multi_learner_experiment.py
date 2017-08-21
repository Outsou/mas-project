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


def create_multi_graphs(folder, window_size):
    # Percentage arrays
    sgd = None
    bandit = None
    linear = None
    length = None
    max_rewards = None

    # Calculate averages from all runs

    for file in os.listdir(folder):
        stats = pickle.load(open(os.path.join(folder, file), 'rb'))

        if sgd is None:
            length = len(stats['max_rewards'])
            sgd = np.zeros(length)
            bandit = np.zeros(length)
            linear = np.zeros(length)
            max_rewards = np.zeros(length)

        sgd = np.add(sgd, stats['sgd']['rewards'])
        bandit = np.add(bandit, stats['bandit']['rewards'])
        linear = np.add(linear, stats['linear']['rewards'])
        max_rewards = np.add(max_rewards, stats['max_rewards'])

    num_of_files = len(os.listdir(folder))
    sgd = sgd / num_of_files
    bandit = bandit / num_of_files
    linear = linear / num_of_files
    max_rewards = max_rewards / num_of_files

    # Use the averages to create a graph

    x = []
    last_idx = length - 1

    sgd_sums = []
    bandit_sums = []
    linear_sums = []


    i = 0
    while i < last_idx:
        # Don't use non-complete window
        if i + window_size - 1 > last_idx:
            break

        sgd_reward = np.sum(sgd[0:i+window_size])
        bandit_reward = np.sum(bandit[0:i+window_size])
        linear_reward = np.sum(linear[0:i+window_size])
        max_reward = np.sum(max_rewards[0:i+window_size])

        sgd_sums.append(sgd_reward / max_reward)
        bandit_sums.append(bandit_reward / max_reward)
        linear_sums.append(linear_reward / max_reward)

        i += window_size

        x.append(i)

    # Draw the graph

    plt.plot(x, sgd_sums, label='SGD')
    plt.plot(x, linear_sums, label='linear')
    plt.plot(x, bandit_sums, label='bandit')
    plt.ylabel('Reward percentage')
    plt.legend()
    plt.show()


if __name__ == "__main__":
    # Parameters
    num_of_agents = 6
    num_of_features = 3
    std = 0.2
    search_width = 100

    num_of_simulations = 100
    num_of_steps = 1500

    create_kwargs = {'length': num_of_features}

    # Environment and simulation

    log_folder = 'multi_test_logs'
    shutil.rmtree(log_folder, ignore_errors=True)

    for _ in range(num_of_simulations):

        menv = create_environment(num_of_slaves=8)

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

    create_multi_graphs(active_folder, 10)

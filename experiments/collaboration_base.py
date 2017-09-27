"""Simple collaboration tests with external agent modeling.
"""

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
import itertools
import operator
import pprint


# Default values for simulation parameters
DEFAULT_PARAMS = {
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


def calculate_averages(folder, models):
    """Calculates average stats from stat files in folder.
    """
    def calculate_same_picks(combinations, stats):

        for combination in combinations:
            picks1 = stats[combination[0]]['connections']
            picks2 = stats[combination[1]]['connections']
            np.sum(np.equal(picks1, picks2))

    keys_to_avg = ['random_rewards', 'max_rewards']

    for model in models:
        keys_to_avg.append((model, 'rewards'))
        keys_to_avg.append((model, 'chose_best'))

    model_combinations = list(itertools.combinations(models, 2))

    files = os.listdir(folder)
    first_stats = pickle.load(open(os.path.join(folder, files[0]), 'rb'))
    avg_mul = 1 / len(files)
    avg_stats = {}
    avg_stats['same_picks'] = {}
    avg_stats['best_pick_mat'] = np.zeros((len(models), ) * 2)

    # Initialize avg stats with the first stat file
    for key in keys_to_avg:
        if type(key) is tuple:
            if key[0] not in avg_stats:
                avg_stats[key[0]] = {}
            avg_stats[key[0]][key[1]] = np.array(first_stats[key[0]][key[1]]) * avg_mul
        else:
            avg_stats[key] = np.array(first_stats[key]) * avg_mul

    # Calculate how many common choices were made by all models
    choices = []
    for model in models:
        choices.append(first_stats[model]['connections'])
    all_choices = list(zip(*choices))
    common_choice_count = np.sum(list(map(lambda x: len(set(x)) == 1, all_choices)))
    avg_stats['same_picks']['all'] = common_choice_count * avg_mul

    # Calculate how many common choices were made by model pairs
    for combination in model_combinations:
        picks1 = np.array(first_stats[combination[0]]['connections'])
        picks2 = np.array(first_stats[combination[1]]['connections'])
        avg_stats['same_picks']['/'.join(combination)] = np.sum(picks1 == picks2) * avg_mul

    # Calculate probability matrix for best picks
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                avg_stats['best_pick_mat'][i, j] = 1
            else:
                chose_best1 = np.array(first_stats[models[i]]['chose_best'])
                chose_best2 = np.array(first_stats[models[j]]['chose_best'])
                both_best_count = np.sum(chose_best1 & chose_best2 == 1)
                avg_stats['best_pick_mat'][i, j] = both_best_count / np.sum(chose_best1) * avg_mul

    # Add the rest of the files to avg_stats
    for i in range(1, len(files)):
        stats = pickle.load(open(os.path.join(folder, files[i]), 'rb'))

        for key in keys_to_avg:
            if type(key) is tuple:
                avg_stats[key[0]][key[1]] += np.array(stats[key[0]][key[1]]) * avg_mul
            else:
                avg_stats[key] += np.array(stats[key]) * avg_mul

        # Calculate how many common choices were made by all models
        choices = []
        for model in models:
            choices.append(stats[model]['connections'])
        all_choices = list(zip(*choices))
        common_choice_count = np.sum(list(map(lambda x: len(set(x)) == 1, all_choices)))
        avg_stats['same_picks']['all'] += common_choice_count * avg_mul

        # Calculate how many common choices were made by model pairs
        for combination in model_combinations:
            picks1 = np.array(stats[combination[0]]['connections'])
            picks2 = np.array(stats[combination[1]]['connections'])
            avg_stats['same_picks']['/'.join(combination)] += np.sum(picks1 == picks2) * avg_mul

        # Calculate probability matrix for best picks
        for i in range(len(models)):
            for j in range(len(models)):
                if not i == j:
                    chose_best1 = np.array(stats[models[i]]['chose_best'])
                    chose_best2 = np.array(stats[models[j]]['chose_best'])
                    both_best_count = np.sum(chose_best1 & chose_best2 == 1)
                    avg_stats['best_pick_mat'][i, j] += both_best_count / np.sum(chose_best1) * avg_mul

    return avg_stats


def create_graphs(folder, window_size, title, file_name, stats, models, draw_windows=False):
    def create_graph(models, maximums, ylabel, title, path, random=None):
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
        plt.savefig(os.path.join(path, file_name))
        plt.close()

    def create_pick_graph(stats, path):
        keys = []
        values = []
        for key, value in sorted(stats['same_picks'].items(), key=operator.itemgetter(0)):
            keys.append(key)
            values.append(value)

        ind = np.arange(len(values))
        width = 0.35
        fig, ax = plt.subplots()
        rects = ax.bar(ind, values, width)
        ax.set_xticks(ind)
        ax.set_xticklabels(keys)
        #plt.show()
        name_split = file_name.split('.')
        name = name_split[0] + '_picks'
        plt.savefig(os.path.join(path, name))
        plt.close()

    def create_best_pick_matrix(stats, path):
        mat = np.around(stats['best_pick_mat'], 2)
        ax = plt.subplot(111, frame_on=False)
        ax.xaxis.set_visible(False)
        ax.yaxis.set_visible(False)
        ax.table(cellText=mat, rowLabels=models, colLabels=models, loc='center')
        name_split = file_name.split('.')
        name = name_split[0] + '_best_mat'
        plt.savefig(os.path.join(path, name), bbox_inches='tight')
        plt.close()

    # Create reward graph
    graph_models = [('SGD', stats['sgd']['rewards']),
              ('Q', stats['bandit']['rewards']),
              ('linear', stats['linear']['rewards'])]
              #('poly', stats['poly']['rewards'])]

    path = os.path.split(folder)[0]

    create_graph(graph_models,
                 stats['max_rewards'],
                 'Reward percentage',
                 title,
                 path,
                 stats['random_rewards'])

    create_pick_graph(stats, path)

    create_best_pick_matrix(stats, path)

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


def create_img_name(replace_params):
    """Create image name for a run where ``replace_params`` is a dictionary
    of parameters that are changed from the defaults.
    """
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


def _get_default_params(params):
    if params is None:
        params = DEFAULT_PARAMS
    else:
        for key, value in DEFAULT_PARAMS.items():
            if key not in params:
                params[key] = value
    return params


def create_agents(menv, params, log_folder, path, create_kwargs):
    rets = []
    for _ in range(params['agents']):
        rules = []

        for i in range(params['features']):
            rules.append(RuleLeaf(DummyFeature(i),
                                  GaussianMapper(np.random.rand(),
                                                 params['std'])))

        rule_weights = []
        for _ in range(len(rules)):
            rule_weights.append(np.random.random())

        task = menv.spawn('agents:MultiAgent',
                          log_folder=log_folder,
                          data_folder=path,
                          artifact_cls=DummyArtifact,
                          create_kwargs=create_kwargs,
                          rules=rules[:params['common_features']],
                          rule_weights=rule_weights[:params['common_features']],
                          std=params['std'],
                          active=True,
                          search_width=params['search_width'],
                          reg_weight=params['reg_weight'])
        ret = aiomas.run(until=task)
        rets.append(ret)
    return rets


def run_experiment(params, num_of_simulations, num_of_steps,
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

    # Get default parameter values for keys that are not present in
    # 'params' dictionary, construct replacement parameters for the experiment
    # matrix and save initial parameters.
    params = _get_default_params(params)
    loop, looping_params = make_loop_matrix(params)
    old_params = params.copy()
    log_folder = None
    avgs_folder = _init_data_folder(data_folder)

    run_id = 0
    times = []

    for replace_params in loop:
        params.update(replace_params)
        create_kwargs = {'length': params['features']}
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
            create_agents(menv, params, log_folder, path, create_kwargs)

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
            for i in range(num_of_steps):
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

                print("Run took {:.3f} seconds. Estimated end time at: {}\n"
                      .format(total_time, est_end_time))

        avg_stats = calculate_averages(path, models)

        title = "Connections: {}, features: {}, search width: {}, runs: {}"\
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


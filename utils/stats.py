import itertools
import operator
import matplotlib.pyplot as plt
import os
import pickle
import numpy as np


def _average_arrays(stats, shape, key, sub_key=None):
    avg = np.zeros(shape)
    for stat in stats:
        if sub_key is None:
            avg += np.array(stat[key])
        else:
            avg += np.array(stat[key][sub_key])
    return avg / len(stats)


def calculate_averages_single(folder):
    files = os.listdir(folder)
    avg_stats = {}
    stats = []
    for file in files:
        stat = pickle.load(open(os.path.join(folder, file), 'rb'))

        # Calculate critique sums
        stat['critique_sums'] = []
        for step_critiques in stat['critiques']:
            stat['critique_sums'].append(sum(eval for _, eval in step_critiques))

        stats.append(stat)

    for key, value in stats[0].items():
        if type(value) == list and all(isinstance(x, (float, int)) for x in value):
            avg_stats[key] = _average_arrays(stats, np.array(value).shape, key)

    return avg_stats


def calculate_averages(folder, models):
    '''Calculates average stats from stat files in folder.'''
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
    avg_stats['wrong_pick_mat'] = np.zeros((len(models), ) * 2)

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

    # Calculate probability matrix for best and not best picks
    for i in range(len(models)):
        for j in range(len(models)):
            if i == j:
                avg_stats['best_pick_mat'][i, j] = 1
                avg_stats['wrong_pick_mat'][i, j] = 0
            else:
                chose_best1 = np.array(first_stats[models[i]]['chose_best']).astype(bool)
                chose_best2 = np.array(first_stats[models[j]]['chose_best']).astype(bool)
                both_best_count = np.sum(chose_best1 & chose_best2)
                avg_stats['best_pick_mat'][i, j] = both_best_count / np.sum(chose_best1) * avg_mul
                wrong_best_count = np.sum(~chose_best1 & chose_best2)
                avg_stats['wrong_pick_mat'][i, j] = wrong_best_count / np.sum(~chose_best1) * avg_mul

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
                    chose_best1 = np.array(stats[models[i]]['chose_best']).astype(bool)
                    chose_best2 = np.array(stats[models[j]]['chose_best']).astype(bool)
                    both_best_count = np.sum(chose_best1 & chose_best2 == 1)
                    avg_stats['best_pick_mat'][i, j] += both_best_count / np.sum(chose_best1) * avg_mul
                    wrong_best_count = np.sum(~chose_best1 & chose_best2)
                    avg_stats['wrong_pick_mat'][i, j] += wrong_best_count / np.sum(~chose_best1) * avg_mul

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
    graph_models = [
        ('Q', stats['bandit']['rewards']),
        ('linear', stats['linear']['rewards'])
    ]
              #('poly', stats['poly']['rewards'])]

    if 'sgd' in stats:
        graph_models.append(('SGD', stats['sgd']['rewards']))

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


def calculate_agent_averages(path, models):
    """Calculates average stats of all the runs in a setup for each agent."""
    avgs = []
    for dir in os.listdir(path):
        directory = os.path.join(path, dir)
        if models is not None:
            avg_stats = calculate_averages(directory, models)
        else:
            avg_stats = calculate_averages_single(directory)
        pickle.dump(avg_stats, open(os.path.join(directory, 'avg_stats.p'), 'wb'))
        avgs.append(avg_stats)
    return avgs


def calculate_avg_of_avgs(avgs):
    """Calculates average stats from multiple average stats."""
    avg_of_avgs = {}

    for key, value in avgs[0].items():
        if type(value) is dict:
            avg_of_avgs[key] = {}
            for sub_key in value.keys():
                avg_of_avgs[key][sub_key] = _average_arrays(avgs,
                                                            avgs[0][key][sub_key].shape,
                                                            key,
                                                            sub_key)
        else:
            avg_of_avgs[key] = _average_arrays(avgs, avgs[0][key].shape, key)

    return avg_of_avgs


def calculate_setup_averages(path, models=None):
    """For each setup stats of all agents are averaged.
    Returns a list containing average stats from each setup."""
    avgs = []
    # Get the directories where stats are saved for each setup
    dirs = [file for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]
    for dir in sorted(dirs):
        setup_avgs = calculate_agent_averages(os.path.join(path, dir), models)
        setup_avg = calculate_avg_of_avgs(setup_avgs)
        avgs.append(setup_avg)
    return avgs


def reward_graph(stats, window_size, filepath):
    rewards = stats['rewards']
    x = list(range(0, len(rewards), window_size))
    rewards_y = [np.sum(rewards[i:i + window_size]) for i in x]

    random_rewards = stats['random_rewards']
    random_y = [np.sum(random_rewards[i:i + window_size]) for i in x]

    plt.plot(x, rewards_y, label='rewards')
    plt.plot(x, random_y, label='random')
    plt.legend()
    # plt.show()
    plt.savefig(filepath)
    plt.close()

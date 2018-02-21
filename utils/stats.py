import itertools
import operator
import os
import pickle
import numpy as np
import re
from experiments.collab.chk_runs import get_last_lines
from tabulate import tabulate
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import table
import pandas as pd
from utils.plot_styling import MODEL_STYLES, MODEL_ORDER, BASE_FIG_SIZE


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
    plt.savefig(os.path.join(save_folder, '{}.pdf'.format(param_name)))
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

def get_dirs_in_dir(path):
    dirs = [os.path.join(path, file) for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]
    return dirs

def common_society_analysis(dirs, pickle_name):
    stat_dict = {}

    evals = []
    novs = []
    vals = []
    for dir in dirs:
        pkl = os.path.join(dir, pickle_name)
        pkl_dict = pickle.load(open(pkl, 'rb'))
        keys = pkl_dict.keys()
        for key in keys:
            agents = list(pkl_dict[key].keys())
            agents.remove('creator')
            for agent in agents:
                evals.append(pkl_dict[key][agent][0])
                novs.append(pkl_dict[key][agent][1]['novelty'])
                vals.append(pkl_dict[key][agent][1]['value'])

    stat_dict['avg_eval'] = sum(evals) / len(evals)
    stat_dict['avg_nov'] = sum(novs) / len(novs)
    stat_dict['avg_val'] = sum(vals) / len(vals)

    return stat_dict


def collab_success_per_step(collab_eval_keys, collab_steps):
    successes = np.zeros(collab_steps)
    for key in collab_eval_keys:
        step = int(key[:5])
        idx = int(step / 2 - 1)
        successes[idx] += 1
    return successes


def get_step_top_values(collab_evals, pref_lists, collab_steps, other_vals=False):
    """Makes a dictionary where values are listed for each step for collab artifacts in collab_evals."""
    top_k = [1, 3, 5]

    top = {'all': {'eval': [[] for x in range(collab_steps)],
                   'val': [[] for x in range(collab_steps)],
                   'nov': [[] for x in range(collab_steps)]}}
    aesthetic_top = {}

    for k in top_k:
        top[str(k)] = {'eval': [[] for x in range(collab_steps)],
                       'val': [[] for x in range(collab_steps)],
                       'nov': [[] for x in range(collab_steps)]}

    for collab_art, stats in collab_evals.items():
        step = int(collab_art[:5])
        idx = int(step / 2 - 1)
        creators = stats['creator'].split(' - ')
        initializer = creators[0]
        other = creators[1]

        if other_vals:
            val_agent = other
            other_agent = initializer
        else:
            val_agent = initializer
            other_agent = other

        eval = collab_evals[collab_art][val_agent][0]
        val = collab_evals[collab_art][val_agent][1]['value']
        nov = collab_evals[collab_art][val_agent][1]['novelty']

        other_eval = collab_evals[collab_art][other_agent][0]
        other_val = collab_evals[collab_art][other_agent][1]['value']
        other_nov = collab_evals[collab_art][other_agent][1]['novelty']

        pref_list = pref_lists[initializer][idx]
        aest = collab_evals[collab_art][val_agent][1]['aesthetic']


        if aest not in aesthetic_top:
            aesthetic_top[aest] = {}
            for k in top_k:
                aesthetic_top[aest][str(k)] = {'eval': [],
                                               'val': [],
                                               'nov': []}
            aesthetic_top[aest]['all'] = {'eval': [],
                                          'val': [],
                                          'nov': []}

        for k in top_k:
            if other in pref_list[:k]:
                top[str(k)]['eval'][idx].append(eval)
                top[str(k)]['val'][idx].append(val)
                top[str(k)]['nov'][idx].append(nov)

                aesthetic_top[aest][str(k)]['eval'].append(eval)
                aesthetic_top[aest][str(k)]['val'].append(val)
                aesthetic_top[aest][str(k)]['nov'].append(nov)

        top['all']['eval'][idx].append(eval)
        top['all']['val'][idx].append(val)
        top['all']['nov'][idx].append(nov)
        # top['all']['eval'][idx].append(other_eval)
        # top['all']['val'][idx].append(other_val)
        # top['all']['nov'][idx].append(other_nov)

        aesthetic_top[aest]['all']['eval'].append(eval)
        aesthetic_top[aest]['all']['val'].append(val)
        aesthetic_top[aest]['all']['nov'].append(nov)
        aesthetic_top[aest]['all']['eval'].append(other_eval)
        aesthetic_top[aest]['all']['val'].append(other_val)
        aesthetic_top[aest]['all']['nov'].append(other_nov)
    return top, aesthetic_top


def append_top_dictionaries(dict1, dict2):
    """Appends top pick stat dictionaries."""
    if len(list(dict1.keys())) == 0:
        return dict2

    res_dict = {}
    for key in dict1.keys():
        res_dict[key] = {}
        for sub_key in dict1[key].keys():
            res_dict[key][sub_key] = []
            for i in range(len(dict1[key][sub_key])):
                res_dict[key][sub_key].append(dict1[key][sub_key][i] + dict2[key][sub_key][i])
    return res_dict


def append_aest_top_dicts(dict1, dict2):
    if len(list(dict1.keys())) == 0:
        return dict2

    res_dict = {}
    for aest in dict1.keys():
        res_dict[aest] = {}
        for k in dict1[aest].keys():
            res_dict[aest][k] = {}
            for key in dict1[aest][k].keys():
                res_dict[aest][k][key] = dict1[aest][k][key] + dict2[aest][k][key]

    return res_dict

def calculate_top_dict_means(top_dict):
    """Calculates stepwise means and overall means."""
    res_dict = {}
    for key in top_dict.keys():
        res_dict[key] = {}
        for sub_key, val_list in top_dict[key].items():
            res_dict[key][sub_key] = []
            all = []
            for i in range(len(top_dict[key][sub_key])):
                if len(top_dict[key][sub_key][i]) == 0:
                    res_dict[key][sub_key].append(None)
                else:
                    res_dict[key][sub_key].append(np.mean(top_dict[key][sub_key][i]))
                all += top_dict[key][sub_key][i]
            res_dict[key][sub_key + '_mean'] = np.mean(all)
    return res_dict


def calculate_aest_top_means(aest_dict):
    mean_dict = {}
    for aest in aest_dict.keys():
        mean_dict[aest] = {}
        for k in aest_dict[aest].keys():
            mean_dict[aest][k] = {}
            for key in aest_dict[aest][k].keys():
                mean_dict[aest][k][key] = np.mean(aest_dict[aest][k][key])
    return mean_dict


def analyze_collab_evals(dirs):
    pickle_name = 'collab_evals.pkl'
    collab_eval_stats = common_society_analysis(dirs, pickle_name)

    # Get number of collab attempts
    first_dir = os.path.split(dirs[0])[1]
    collab_iters = int(int(re.findall(r'i\d+', first_dir)[0][1:]) / 2)
    agents = int(re.findall(r'a\d+', first_dir)[0][1:])

    collab_attempts = agents / 2 * collab_iters
    collab_ratios = []

    aesthetic_pair_separate_vals = {}
    aesthetic_pair_vals = {}
    top_pick_stats = {}
    top_pick_stats_other = {}
    aesthetic_top_stats = {}
    aesthetic_top_stats_other = {}
    step_successes = np.zeros(collab_iters)
    for dir in dirs:
        # Load pickles
        collab_evals_pkl = os.path.join(dir, pickle_name)
        collab_evals = pickle.load(open(collab_evals_pkl, 'rb'))
        pref_lists_pkl = os.path.join(dir, 'pref_lists.pkl')
        pref_lists = pickle.load(open(pref_lists_pkl, 'rb'))
        collab_arts = collab_evals.keys()

        # Calculate per step successes
        step_successes += collab_success_per_step(list(collab_evals.keys()), collab_iters)

        # Calculate per step top pick values
        top_picks, aesthetic_tops = get_step_top_values(collab_evals, pref_lists, collab_iters)
        top_picks_other, aesthetic_tops_other = get_step_top_values(collab_evals, pref_lists, collab_iters, True)

        top_pick_stats = append_top_dictionaries(top_pick_stats, top_picks)
        aesthetic_top_stats = append_aest_top_dicts(aesthetic_top_stats, aesthetic_tops)

        top_pick_stats_other = append_top_dictionaries(top_pick_stats_other, top_picks_other)
        aesthetic_top_stats_other = append_aest_top_dicts(aesthetic_top_stats_other, aesthetic_tops_other)

        for collab_art in collab_arts:
            creators = collab_evals[collab_art]['creator'].split(' - ')

            # Calculate averages for aesthetic pairs
            pair = []
            for creator in creators:
                pair.append(collab_evals[collab_art][creator][1]['aesthetic'])
            pair.sort()
            pair = tuple(pair)
            if pair not in aesthetic_pair_vals:
                aesthetic_pair_vals[pair] = {'eval': [],
                                             'val': [],
                                             'nov': [],
                                             'count': 0}
                aesthetic_pair_separate_vals[pair] = {pair[0]: [],
                                                      pair[1]: []}

            for creator in creators:
                val = collab_evals[collab_art][creator][1]['value']
                aest = collab_evals[collab_art][creator][1]['aesthetic']
                aesthetic_pair_vals[pair]['eval'].append(collab_evals[collab_art][creator][0])
                aesthetic_pair_vals[pair]['val'].append(val)
                aesthetic_pair_vals[pair]['nov'].append(collab_evals[collab_art][creator][1]['novelty'])
                aesthetic_pair_separate_vals[pair][aest].append(val)

            aesthetic_pair_vals[pair]['count'] += 1

            # Calculate collab success ratio for simulation run
            collab_ratios.append(len(collab_arts) / collab_attempts)


    collab_eval_stats['success_ratio'] = {'mean': np.mean(collab_ratios),
                                          'conf_int': st.t.interval(0.99, len(collab_ratios) - 1,
                                                                    loc=np.mean(collab_ratios),
                                                                    scale=st.sem(collab_ratios))}

    collab_eval_stats['aesthetic_pairs'] = {}
    for pair, vals in aesthetic_pair_vals.items():
        collab_eval_stats['aesthetic_pairs'][pair] = {'eval': sum(vals['eval']) / len(vals['eval']),
                                                      'val': sum(vals['val']) / len(vals['val']),
                                                      'nov': sum(vals['nov']) / len(vals['nov']),
                                                      'count': vals['count']}

    for pair, aests in aesthetic_pair_separate_vals.items():
        for aest, vals in aests.items():
            aesthetic_pair_separate_vals[pair][aest] = np.mean(aesthetic_pair_separate_vals[pair][aest])
    collab_eval_stats['aest_pair_separate_vals'] = aesthetic_pair_separate_vals

    top_pick_means = calculate_top_dict_means(top_pick_stats)
    aesthetic_top_pick_means = calculate_aest_top_means(aesthetic_top_stats)

    top_pick_means_other = calculate_top_dict_means(top_pick_stats_other)
    aesthetic_top_pick_means_other = calculate_aest_top_means(aesthetic_top_stats_other)

    collab_eval_stats['top_pick_stats'] = top_pick_means
    collab_eval_stats['aest_top_pick_stats'] = aesthetic_top_pick_means
    collab_eval_stats['top_pick_stats_other'] = top_pick_means_other
    collab_eval_stats['aest_top_pick_stats_other'] = aesthetic_top_pick_means_other
    collab_eval_stats['step_successes'] = step_successes / len(dirs)
    collab_eval_stats['num_of_agents'] = agents

    return collab_eval_stats


def analyze_ind_evals(dirs):
    pickle_name = 'ind_evals.pkl'
    ind_eval_stats = common_society_analysis(dirs, pickle_name)

    agents = None
    aest_agents = {}

    aest_vals = {}

    for dir in dirs:
        pkl = os.path.join(dir, pickle_name)
        ind_evals = pickle.load(open(pkl, 'rb'))
        for art in ind_evals.keys():
            # Create agent list and a dictionary with an agent for each aesthetic
            if agents is None:
                agents = list(ind_evals[art].keys())
                agents.remove('creator')
                aest_agents = {}
                for agent in agents:
                    agent_aest = ind_evals[art][agent][1]['aesthetic']
                    if agent_aest not in aest_agents:
                        aest_agents[agent_aest] = agent

            # Get val for each aesthetic
            creator = ind_evals[art]['creator']
            creator_aest = ind_evals[art][creator][1]['aesthetic']
            if creator_aest not in aest_vals:
                aest_vals[creator_aest] = {}
            for aest in aest_agents.keys():
                if aest != creator_aest:
                    if aest not in aest_vals[creator_aest]:
                        aest_vals[creator_aest][aest] = []
                    agent = aest_agents[aest]
                    val = ind_evals[art][agent][1]['value']
                    aest_vals[creator_aest][aest].append(val)

    for aest1, aests in aest_vals.items():
        for aest2, vals in aests.items():
            aest_vals[aest1][aest2] = np.mean(vals)
    ind_eval_stats['aest_vals'] = aest_vals

    return ind_eval_stats


def common_agent_analysis(dirs, pkl_name):
    stat_dict = {}

    evals = []
    novs = []
    vals = []
    for dir in dirs:
        sub_dirs = get_dirs_in_dir(dir)
        for sub_dir in sub_dirs:
            pkl = os.path.join(sub_dir, pkl_name)
            pkl_dict = pickle.load(open(pkl, 'rb'))
            evals += pkl_dict['eval']
            novs += pkl_dict['nov']
            vals += pkl_dict['val']

    stat_dict['avg_eval'] = sum(evals) / len(evals)
    stat_dict['avg_nov'] = sum(novs) / len(novs)
    stat_dict['avg_val'] = sum(vals) / len(vals)
    return stat_dict


def analyze_collab_arts(dirs):
    pkl_name = 'collab_arts.pkl'
    collab_art_stats = {}

    aesthetic_pair_stats = {}
    aesthetic_stats = {}

    for dir in dirs:
        sub_dirs = get_dirs_in_dir(dir)
        for sub_dir in sub_dirs:
            collab_arts_dict = pickle.load(open(os.path.join(sub_dir, pkl_name), 'rb'))
            general_info = pickle.load(open(os.path.join(sub_dir, 'general_info.pkl'), 'rb'))

            # Calculate how this aesthetic does with other aesthetics
            aest = general_info['aesthetic']
            if aest not in aesthetic_stats:
                aesthetic_stats[aest] = {}

            for i in range(len(collab_arts_dict['fb'])):
                caest = collab_arts_dict['caest'][i]
                if caest not in aesthetic_stats[aest]:
                    aesthetic_stats[aest][caest] = {'eval': [],
                                                    'val': [],
                                                    'nov': []}
                # If collab succeeded
                if collab_arts_dict['fb'][i]:
                    # Calculate index in the value lists
                    idx = sum(collab_arts_dict['fb'][:i+1]) - 1
                    aesthetic_stats[aest][caest]['eval'].append(collab_arts_dict['eval'][idx])
                    aesthetic_stats[aest][caest]['val'].append(collab_arts_dict['val'][idx])
                    aesthetic_stats[aest][caest]['nov'].append(collab_arts_dict['nov'][idx])

            # Calculate how often aesthetic pairs succeeded and failed
            for i in range(len(collab_arts_dict['fb'])):
                caest = collab_arts_dict['caest'][i]
                aest_pair = tuple(sorted([aest, caest]))
                if aest_pair not in aesthetic_pair_stats:
                    aesthetic_pair_stats[aest_pair] = {'succeeded': 0, 'failed': 0, 'rank': [],
                                                       'succ_init': {aest_pair[0]: 0, aest_pair[1]: 0}}
                if collab_arts_dict['fb'][i]:
                    aesthetic_pair_stats[aest_pair]['succeeded'] += 1
                    initializer = aest if collab_arts_dict['cinit'][i] else caest
                    aesthetic_pair_stats[aest_pair]['succ_init'][initializer] += 1
                    # Calculate index of rank
                    rank_idx = sum(collab_arts_dict['fb'][:i+1]) - 1
                    aesthetic_pair_stats[aest_pair]['rank'].append(collab_arts_dict['rank'][rank_idx])
                else:
                    aesthetic_pair_stats[aest_pair]['failed'] += 1

    # Divide aesthetic pair stats by 2, because the same stat is calculated for both agents
    for aest_pair in aesthetic_pair_stats.keys():
        aesthetic_pair_stats[aest_pair]['succeeded'] /= 2
        aesthetic_pair_stats[aest_pair]['failed'] /= 2
        aesthetic_pair_stats[aest_pair]['rank'] = sum(aesthetic_pair_stats[aest_pair]['rank']) \
                                                  / len(aesthetic_pair_stats[aest_pair]['rank'])
        for key in aesthetic_pair_stats[aest_pair]['succ_init'].keys():
            aesthetic_pair_stats[aest_pair]['succ_init'][key] = \
                int(aesthetic_pair_stats[aest_pair]['succ_init'][key] / 2)

    # Calculate aesthetic averages
    for aesthetic, caesthetics in aesthetic_stats.items():
        for caesthetic, vals in caesthetics.items():
            aesthetic_stats[aesthetic][caesthetic]['eval'] = sum(vals['eval']) / len(vals['eval'])
            aesthetic_stats[aesthetic][caesthetic]['val'] = sum(vals['val']) / len(vals['val'])
            aesthetic_stats[aesthetic][caesthetic]['nov'] = sum(vals['nov']) / len(vals['nov'])

    collab_art_stats['aesthetic_pairs'] = aesthetic_pair_stats
    collab_art_stats['aesthetic'] = aesthetic_stats

    return collab_art_stats


def count_first(pref_lists, addr):
    count = 0
    for key, lists in pref_lists.items():
        if key != addr:
            for pref_list in lists:
                if addr == pref_list[0]:
                    count += 1
    return count


def analyze_own_arts(dirs):
    own_art_stats = common_agent_analysis(dirs, 'own_arts.pkl')
    skill_dict = {}
    first_dir = os.path.split(dirs[0])[1]
    iters = int(int(re.findall(r'i\d+', first_dir)[0][1:]) / 2)

    vals = [[] for _ in range(iters)]
    novs = [[] for _ in range(iters)]
    aest_stats = {}

    for dir in dirs:
        pref_lists = pickle.load(open(os.path.join(dir, 'pref_lists.pkl'), 'rb'))
        sub_dirs = get_dirs_in_dir(dir)
        for sub_dir in sub_dirs:
            pkl = os.path.join(sub_dir, 'own_arts.pkl')
            own_arts = pickle.load(open(pkl, 'rb'))
            general_info = pickle.load(open(os.path.join(sub_dir, 'general_info.pkl'), 'rb'))
            aest = general_info['aesthetic']

            num_of_skills = len(general_info['pset_names'])
            if num_of_skills not in skill_dict:
                skill_dict[num_of_skills] = {}
                skill_dict[num_of_skills]['count'] = 0
                skill_dict[num_of_skills]['val'] = []

            skill_dict[num_of_skills]['count'] += count_first(pref_lists, general_info['addr'])

            if aest not in aest_stats:
                aest_stats[aest] = {'evals': [],
                                    'vals': [],
                                    'novs': []}

            aest_stats[aest]['evals'] += own_arts['eval']
            aest_stats[aest]['vals'] += own_arts['val']
            aest_stats[aest]['novs'] += own_arts['nov']

            skill_dict[num_of_skills]['val'] += own_arts['val']

            for i in range(iters):
                vals[i].append(own_arts['val'][i])
                novs[i].append(own_arts['nov'][i])

    for i in range(iters):
        vals[i] = np.mean(vals[i])
        novs[i] = np.mean(novs[i])

    aest_means = {}
    for aest in aest_stats.keys():
        aest_means[aest] = {}
        aest_means[aest]['eval'] = np.mean(aest_stats[aest]['evals'])
        aest_means[aest]['val'] = np.mean(aest_stats[aest]['vals'])
        aest_means[aest]['nov'] = np.mean(aest_stats[aest]['novs'])

    for skill_amount in skill_dict.keys():
        skill_dict[skill_amount]['val'] = np.mean(skill_dict[skill_amount]['val'])

    own_art_stats['step_vals'] = vals
    own_art_stats['step_novs'] = novs
    own_art_stats['aest_means'] = aest_means
    own_art_stats['skill_dict'] = skill_dict
    return own_art_stats


def analyze_model_dir(path):
    # Get directories of valid runs
    dirs = []
    lines = get_last_lines(path)
    for line in lines:
        if 'Run finished' in line:
            dirs.append(line.split(': ')[0])

    collab_eval_stats = analyze_collab_evals(dirs)
    collab_art_stats = analyze_collab_arts(dirs)
    own_art_stats = analyze_own_arts(dirs)
    ind_eval_stats = analyze_ind_evals(dirs)

    return collab_eval_stats, collab_art_stats, own_art_stats, ind_eval_stats

def make_table_image(mat, row_name, col_names, filename, size=(12, 4)):
    df = pd.DataFrame(mat, index=row_name, columns=col_names)
    fig, ax = plt.subplots(figsize=size)  # set size frame
    ax.xaxis.set_visible(False)  # hide the x axis
    ax.yaxis.set_visible(False)  # hide the y axis
    ax.set_frame_on(False)  # no visible frame, uncomment if size is ok
    tabla = table(ax, df, loc='upper right', colWidths=[0.17] * len(df.columns))  # where df is your data frame
    tabla.auto_set_font_size(False)  # Activate set fontsize manually
    tabla.set_fontsize(12)  # if ++fontsize is necessary ++colWidths
    tabla.scale(1.2, 1.2)  # change size table
    plt.savefig(filename, transparent=True)
    plt.close()

def get_cumulative_success_ratio(successes, step_attempts):
    """Calculates cumulative success ratio."""
    cum_ratios = []
    for i in range(len(successes)):
        attempts = step_attempts * (i + 1)
        cum_ratios.append(sum(successes[:i+1]) / attempts)
    return cum_ratios

def make_success_ratio_plot(ratio_dict, filename, x=None):
    """Creates a success ratio plot."""
    mratios = sorted(ratio_dict.items(), key=lambda x: MODEL_ORDER.index(x[0]))

    for model, ratios in mratios:
        if x is None:
            steps = len(ratios) * 2
            x = list(range(2, steps + 1, 2))
        style = MODEL_STYLES[model]
        plt.plot(x, ratios, style['line style'],
                     dashes=style['dashes'], label=style['label'], color=style['color'])
    plt.xlabel('step')
    plt.ylabel('success ratio')
    plt.legend()
    fig = plt.gcf()
    fig.set_size_inches(BASE_FIG_SIZE[0], BASE_FIG_SIZE[1])
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def print_aesthetic_table(collab_eval_stats, collab_art_stats, format_s, models):
    """Creates and prints statistic table for aesthetic pairs."""
    aesthetic_pair_rows = []
    pairs = list(collab_eval_stats['aesthetic_pairs'].keys())
    pairs.sort()
    for pair in pairs:
        total = collab_art_stats['aesthetic_pairs'][pair]['succeeded'] \
                + collab_art_stats['aesthetic_pairs'][pair]['failed']
        success_ratio = collab_art_stats['aesthetic_pairs'][pair]['succeeded'] / total
        row_vals = ['{}, {}'.format(pair[0], pair[1]),
                    collab_eval_stats['aesthetic_pairs'][pair]['eval'],
                    collab_eval_stats['aesthetic_pairs'][pair]['val'],
                    collab_eval_stats['aesthetic_pairs'][pair]['nov'],
                    collab_art_stats['aesthetic_pairs'][pair]['succeeded'],
                    total,
                    success_ratio,
                    collab_art_stats['aesthetic_pairs'][pair]['rank'],
                    '{}/{}'.format(collab_art_stats['aesthetic_pairs'][pair]['succ_init'][pair[0]],
                                   collab_art_stats['aesthetic_pairs'][pair]['succ_init'][pair[1]])]
        row_vals = [format_s % val if type(val) is float else val for val in row_vals]
        aesthetic_pair_rows.append(row_vals)

    print(tabulate(aesthetic_pair_rows,
                   headers=[models[-1], 'evaluation', 'value', 'novelty', 'success count', 'total count',
                            'success ratio', 'mean rank', 'successfull inits']))
    return aesthetic_pair_rows, pairs


def print_aesthetic_stats(collab_art_stats, format_s):
    """Creates and prints aesthetic table from single aesthetics point of view."""
    aesthetic_rows = []
    aesthetics = list(collab_art_stats['aesthetic'].keys())
    for aest in sorted(aesthetics):
        for caest in sorted(collab_art_stats['aesthetic'][aest].keys()):
            row = [aest,
                   caest,
                   collab_art_stats['aesthetic'][aest][caest]['eval'],
                   collab_art_stats['aesthetic'][aest][caest]['val'],
                   collab_art_stats['aesthetic'][aest][caest]['nov']]
            aesthetic_rows.append([format_s % val if type(val) in [float, np.float64] else val for val in row])

    print(tabulate(aesthetic_rows,
                   headers=['aesthetic', 'collab aesthetic', 'evaluation', 'value', 'novelty']))


def create_top_k_val_nov_plot(top_pick_stats, model):
    x = list(range(2, len(top_pick_stats['all']['val']) * 2 + 1, 2))

    for val in ['val', 'nov']:
        for key in sorted(top_pick_stats.keys()):
            plt.plot(x, top_pick_stats[key][val], label=key)
        plt.xlabel('step')
        plt.ylabel(val)
        plt.legend()
        plt.ylim(0.4, 0.7)
        fig = plt.gcf()
        fig.set_size_inches(BASE_FIG_SIZE[0], BASE_FIG_SIZE[1])
        plt.tight_layout()
        plt.savefig('{}_{}_top_picks.png'.format(model, val))
        plt.savefig('{}_{}_top_picks.pdf'.format(model, val))
        plt.close()


def create_val_nov_plot(step_pick_stat_dict, mean_solo_vals):
    first_key = list(step_pick_stat_dict.keys())[0]
    x = list(range(2, len(step_pick_stat_dict[first_key]['all']['val']) * 2 + 1, 2))
    models = sorted(step_pick_stat_dict.keys(), key=lambda x: MODEL_ORDER.index(x))
    for val in ['val', 'nov']:
        if val == 'val':
            plt.plot(x[1:], mean_solo_vals[1:], '--o', alpha=0.25, label='Solo mean', markevery=10, color='k')

        for model in models:
            style = MODEL_STYLES[model]
            plt.plot(x, step_pick_stat_dict[model]['all'][val], style['line style'],
                     dashes=style['dashes'], label=style['label'], color=style['color'])

        plt.xlabel('step')
        plt.ylabel(val)
        plt.legend()
        plt.ylim(0.4, 0.7)
        fig = plt.gcf()
        fig.set_size_inches(BASE_FIG_SIZE[0], BASE_FIG_SIZE[1])
        plt.tight_layout()
        plt.savefig('{}_collab.png'.format(val))
        plt.savefig('{}_collab.pdf'.format(val))
        plt.close()


def create_solo_val_nov_plots(step_vals_novs_solo):
    first_key = list(step_vals_novs_solo.keys())[0]
    x = list(range(2, len(step_vals_novs_solo[first_key]['vals']) * 2 + 1, 2))

    models = sorted(step_vals_novs_solo.keys(), key=lambda x: MODEL_ORDER.index(x))

    for val in ['vals', 'novs']:
        for model in models:
            style = MODEL_STYLES[model]
            plt.plot(x, step_vals_novs_solo[model][val], style['line style'],
                     dashes=style['dashes'], label=style['label'], color=style['color'])
        plt.xlabel('step')
        plt.ylabel(val)
        plt.legend()
        fig = plt.gcf()
        fig.set_size_inches(BASE_FIG_SIZE[0], BASE_FIG_SIZE[1])
        plt.tight_layout()
        plt.savefig('{}_solo.png'.format(val))
        plt.savefig('{}_solo.pdf'.format(val))
        plt.close()


def make_aest_val_table_and_image(own_art_stats, ind_eval_stats, pairs, collab_eval_stats, format_s, img_name = None):
    rows = []

    aest1_val1 = []
    aest1_val2 = []
    collab_val1 = []
    collab_val2 = []
    aest2_val1 = []
    aest2_val2 = []

    for pair in pairs:
        if pair[0] != pair[1]:
            if pair[0] == 'benford' and pair[1] == 'fd_aesthetics':
                idx = len(aest1_val1)

            aest1_val1.append(own_art_stats['aest_means'][pair[0]]['val'])
            aest1_val2.append(ind_eval_stats['aest_vals'][pair[0]][pair[1]])
            collab_val1.append(collab_eval_stats['aest_pair_separate_vals'][pair][pair[0]])
            collab_val2.append(collab_eval_stats['aest_pair_separate_vals'][pair][pair[1]])
            aest2_val1.append(ind_eval_stats['aest_vals'][pair[1]][pair[0]])
            aest2_val2.append(own_art_stats['aest_means'][pair[1]]['val'])
            row = [pair,
                   '{}/{}'.format(format_s % aest1_val1[-1],
                                  format_s % aest1_val2[-1]),
                   '{}/{}'.format(format_s % collab_val1[-1],
                                  format_s % collab_val2[-1]),
                   '{}/{}'.format(format_s % aest2_val1[-1],
                                  format_s % aest2_val2[-1])]
            rows.append(row)
    print(tabulate(rows, headers=['1./2.', 'created by 1.', 'created in collab', 'created by 2.']))

    if img_name is not None:
        N = 3
        aest1_vals = (aest1_val1[idx], collab_val1[idx], aest2_val1[idx])

        ind = np.arange(N)  # the x locations for the groups
        width = 0.35  # the width of the bars

        fig, ax = plt.subplots()
        rects1 = ax.bar(ind, aest1_vals, width, color='r')

        aest2_vals = (aest1_val2[idx], collab_val2[idx], aest2_val2[idx])
        rects2 = ax.bar(ind + width, aest2_vals, width, color='y')

        # add some text for labels, title and axes ticks
        ax.set_ylabel('Value')
        ax.set_xlabel('Creator')
        ax.set_xticks(ind + width / 2)
        ax.set_xticklabels(('benford', 'both', 'fd_aesthetics'))

        ax.legend((rects1[0], rects2[0]), ('benford', 'fd_aesthetics'))
        plt.savefig(img_name)
        plt.close()


def make_aesthetic_rows(collab_eval_stats, own_art_stats, aest_rows, aest_first_choice_rows, eval_ratios, format_s):
    for aest in collab_eval_stats['aest_top_pick_stats']:
        if aest not in eval_ratios:
            eval_ratios[aest] = {'rand': [], 'first': [], 'rand_first': [], 'first ': [], 'other': []}

        if aest not in aest_rows:
            aest_rows[aest] = [['{} own collab eval'.format(aest)],
                               ['{} own solo eval'.format(aest)],
                               ['{} own collab val'.format(aest)],
                               ['{} own solo val'.format(aest)]]
                                # ['{} own collab nov'.format(aest)],
                                # ['{} own solo nov'.format(aest)]]
        own_collab_val = collab_eval_stats['aest_top_pick_stats'][aest]['all']['val']
        own_solo_val = own_art_stats['aest_means'][aest]['val']
        col_vals = [collab_eval_stats['aest_top_pick_stats'][aest]['all']['eval'],
                    own_art_stats['aest_means'][aest]['eval'],
                    own_collab_val,
                    own_solo_val]
                    # collab_eval_stats['aest_top_pick_stats'][aest]['all']['nov'],
                    # own_art_stats['aest_means'][aest]['nov']]
        col_vals = [format_s % val if type(val) in [float, np.float64] else val for val in col_vals]
        for i in range(len(col_vals)):
            aest_rows[aest][i].append(col_vals[i])

        if aest not in aest_first_choice_rows:
            aest_first_choice_rows[aest] = [['{} first choice eval'.format(aest)],
                                            ['{} first choice val'.format(aest)]]
                                            # ['{} first choice nov'.format(aest)]]
        first_choice_val = collab_eval_stats['aest_top_pick_stats'][aest]['1']['val']
        col_vals = [collab_eval_stats['aest_top_pick_stats'][aest]['1']['eval'],
                    collab_eval_stats['aest_top_pick_stats'][aest]['1']['val']]
                    # collab_eval_stats['aest_top_pick_stats'][aest]['1']['nov']]
        col_vals = [format_s % val if type(val) in [float, np.float64] else val for val in col_vals]
        for i in range(len(col_vals)):
            aest_first_choice_rows[aest][i].append(col_vals[i])

        eval_ratios[aest]['first'].append(first_choice_val / own_collab_val)
        eval_ratios[aest]['rand'].append(own_collab_val)
        eval_ratios[aest]['rand_first'].append(first_choice_val)
        eval_ratios[aest]['first '].append(first_choice_val / own_solo_val)
        eval_ratios[aest]['other'].append(collab_eval_stats['aest_top_pick_stats_other'][aest]['1']['val'] /
                                          own_solo_val)


def make_pair_count_bar_graph_all(pair_counts):
    proportions = {}

    # Get normalized proportions
    for model, pairs in pair_counts.items():
        proportions[model] = {}
        proportions[model]['failed'] = []
        proportions[model]['succeeded'] = []

        total = 0
        for pair, vals in pairs.items():
            total += vals['succeeded'] + vals['failed']

        pairs_sorted = sorted(pairs.items())

        for _, vals in pairs_sorted:
            proportions[model]['failed'].append(vals['failed'] / total)
            proportions[model]['succeeded'].append(vals['succeeded'] / total)

    # Make the graph
    proportions_sorted = sorted(proportions.items(), key=lambda x: MODEL_ORDER.index(x[0]))
    N = len(proportions_sorted[0][1]['succeeded'])
    ind = np.arange(N)
    width = 0.18
    fig, ax = plt.subplots()
    i = 0

    for model, proportions in proportions_sorted:
        ax.bar(ind + width * i, proportions['succeeded'], width, label=MODEL_STYLES[model]['label'])
        ax.bar(ind + width * i, proportions['failed'], width, bottom=proportions['succeeded'], color='red')
        i += 1
    ax.set_xticks(ind + width * 2)
    ax.set_xticklabels([x[0] for x in pairs_sorted], rotation='vertical')

    # for model, proportions in proportions_sorted:
    #     ax.barh(ind + width * i, proportions['succeeded'], width, label=MODEL_STYLES[model]['label'])
    #     ax.barh(ind + width * i, proportions['failed'], width, left=proportions['succeeded'], color='red')
    #     i += 1
    # ax.set_yticks(ind + width * 2)
    # ax.set_yticklabels([x[0] for x in pairs_sorted])
    # ax.invert_yaxis()

    plt.legend()
    plt.tight_layout()

    plt.show()


def make_pair_count_bar_graph(pair_counts, model):
    failed_counts = []
    succeeded_counts = []

    for pair, counts in sorted(pair_counts.items()):
        failed_counts.append(counts['failed'])
        succeeded_counts.append(counts['succeeded'])

    failed_counts = np.array(failed_counts)
    succeeded_counts = np.array(succeeded_counts)
    total = np.sum(failed_counts) + np.sum(succeeded_counts)
    failed_counts /= total
    succeeded_counts /= total

    fail_success = zip(failed_counts, succeeded_counts)

    counts_sorted = sorted(fail_success, key=lambda x: x[0] + x[1], reverse=True)
    fails_sorted, succeeded_sorted = zip(*counts_sorted)

    ind = np.arange(len(counts_sorted))
    p1 = plt.bar(ind, succeeded_sorted, tick_label='', edgecolor='black')
    p2 = plt.bar(ind, fails_sorted, tick_label='', edgecolor='black', bottom=succeeded_sorted)
    plt.ylabel('Proportion of pair')
    plt.xlabel('Pair')
    plt.ylim(0, 0.18)
    plt.title('{} aesthetic pair proportions.'.format(MODEL_STYLES[model]['label']))
    plt.legend((p1[0], p2[0]), ('succeeded', 'failed'))
    plt.savefig('{}_pair_diversity.png'.format(model))
    plt.savefig('{}_pair_diversity.pdf'.format(model))
    plt.close()


def smooth_success_ratios(success_ratios, window=2):
    random_model = list(success_ratios.keys())[0]
    length = len(success_ratios[random_model])

    smoothed_ratios = {}
    idx = np.array(range(0, length, window))

    for model in success_ratios.keys():
        smoothed_ratios[model] = []
        for i in idx:
            vals = success_ratios[model][i:i+window]
            smoothed_ratios[model].append(np.mean(vals))

    x = idx + window
    if x[-1] >= length:
        x[-1] = length - 1

    return smoothed_ratios, x

def analyze_collab_gp_runs(path, decimals=3, exclude=None):
    """The main function to call when analyzing runs."""
    # random needs to be last
    LIST_ORDER = ['lr', 'Q1', 'Q2', 'Q3', 'hedonic-Q', 'state-Q', 'state-Q2', 'state-Q3', 'random']
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    model_dirs = sorted(get_dirs_in_dir(path), key=lambda x: LIST_ORDER.index(os.path.split(x)[1]))
    # model_dirs = sorted(get_dirs_in_dir(path))
    format_s = '%.{}f'.format(decimals)

    rows = [['Collaboration success ratio'],
            ['Success ratio conf int'],
            ['Evaluation of own collab artifacts'],
            ['Evaluation of own solo artifacts'],
            ['Value of own collab artifacts'],
            ['Initializers value of collab artifacts'],
            ['Partners value of collab artifacts'],
            ['Value of own solo artifacts'],
            #['Novelty of own collab artifacts'],
            #['Novelty of own solo artifacts'],
            ['Overall evaluation of collab artifacts'],
            ['Overall evaluation of solo artifacts'],
            ['Overall value of collab artifacts'],
            ['Overall value of solo artifacts'],
            #['Overall novelty of collab artifacts'],
            #['Overall novelty of solo artifacts'],
            ['First choice collab evaluation'],
            ['First choice collab value'],
            #['First choice collab novelty'],
            ['First choice partner\'s evaluation'],
            ['First choice partner\'s value']]
            #['First choice partner\'s novelty']]
    aest_rows = {}
    aest_first_choice_rows = {}

    models = ['']
    cumulative_successes = {}
    step_success_ratios = {}
    step_pick_stat_dict = {}
    step_vals_novs_solo = {}
    eval_ratios = {}
    pair_counts = {}

    for model_dir in model_dirs:
        model = os.path.split(model_dir)[1]

        if exclude is not None and model in exclude:
            continue

        models.append(model)
        collab_eval_stats, collab_art_stats, own_art_stats, ind_eval_stats = analyze_model_dir(model_dir)

        # Generate aesthetic dependant rows
        make_aesthetic_rows(collab_eval_stats, own_art_stats, aest_rows, aest_first_choice_rows, eval_ratios, format_s)

        conf_int = collab_eval_stats['success_ratio']['conf_int']
        conf_int = (format_s % conf_int[0], format_s % conf_int[1])

        # Add column to main table
        init_val = collab_eval_stats['top_pick_stats']['all']['val_mean']
        partner_val = collab_eval_stats['top_pick_stats_other']['all']['val_mean']
        collab_eval = (collab_eval_stats['top_pick_stats']['all']['eval_mean'] +
                       collab_eval_stats['top_pick_stats_other']['all']['eval_mean']) / 2

        col_vals = [collab_eval_stats['success_ratio']['mean'],
                    (conf_int),
                    collab_eval,
                    own_art_stats['avg_eval'],
                    (init_val + partner_val) / 2,
                    init_val,
                    partner_val,
                    own_art_stats['avg_val'],
                    #collab_eval_stats['top_pick_stats']['all']['nov_mean'],
                    #own_art_stats['avg_nov'],
                    collab_eval_stats['avg_eval'],
                    ind_eval_stats['avg_eval'],
                    collab_eval_stats['avg_val'],
                    ind_eval_stats['avg_val'],
                    #collab_eval_stats['avg_nov'],
                    #ind_eval_stats['avg_nov'],
                    collab_eval_stats['top_pick_stats']['1']['eval_mean'],
                    collab_eval_stats['top_pick_stats']['1']['val_mean'],
                    #collab_eval_stats['top_pick_stats']['1']['nov_mean'],
                    collab_eval_stats['top_pick_stats_other']['1']['eval_mean'],
                    collab_eval_stats['top_pick_stats_other']['1']['val_mean']]
                    #collab_eval_stats['top_pick_stats_other']['1']['nov_mean']]

        col_vals = [format_s % val if type(val) in [float, np.float64] else val for val in col_vals]
        for i in range(len(rows)):
            rows[i].append(col_vals[i])

        # Create and print aesthetic pair table
        aesthetic_pair_rows, pairs = print_aesthetic_table(collab_eval_stats, collab_art_stats, format_s, models)
        print()

        # Print aesthetic pair separate val table
        img_name = None
        if model == 'random':
            img_name = 'random_aest_vals.png'
        # make_aest_val_table_and_image(own_art_stats, ind_eval_stats, pairs, collab_eval_stats, format_s, img_name)
        print()

        # Create image of aesthetic pair count, success ratio and mean rank
        # table = np.array(aesthetic_pair_rows)[:, -3:]
        # table = np.array(table, dtype=float)
        # make_table_image(np.round(table, 2),
        #                  pairs,
        #                  ['count', 'success ratio', 'mean rank'],
        #                  '{}_aest_pairs.png'.format(model))

        # Record success ratios
        cumulative_successes[model] = get_cumulative_success_ratio(collab_eval_stats['step_successes'],
                                                                   collab_eval_stats['num_of_agents'] / 2)
        step_success_ratios[model] = collab_eval_stats['step_successes'] / (collab_eval_stats['num_of_agents'] / 2)

        # Create and print aesthetic stats
        print_aesthetic_stats(collab_art_stats, format_s)
        print()

        # Create value/novelty plot for this model
        create_top_k_val_nov_plot(collab_eval_stats['top_pick_stats'], model)

        step_pick_stat_dict[model] = collab_eval_stats['top_pick_stats']

        step_vals_novs_solo[model] = {}
        step_vals_novs_solo[model]['vals'] = own_art_stats['step_vals']
        step_vals_novs_solo[model]['novs'] = own_art_stats['step_novs']

        # Make pair count bar graph
        make_pair_count_bar_graph(collab_art_stats['aesthetic_pairs'], model)
        pair_counts[model] = collab_art_stats['aesthetic_pairs']

        # Print how many times agents with different amount of skills were chosen
        print('Skills:\tchosen/mean solo val')
        for skill_amount, stats in sorted(own_art_stats['skill_dict'].items()):
            val = format_s % stats['val']
            print('{}:\t\t{}/{}'.format(skill_amount, stats['count'], val))


    # make_pair_count_bar_graph_all(pair_counts)

    # Calculate own collab eval ratios w.r.t random
    for aest in eval_ratios.keys():
        eval_ratios[aest]['rand'] = \
            list(np.array(eval_ratios[aest]['rand']) / eval_ratios[aest]['rand'][-1])
        eval_ratios[aest]['rand_first'] \
            = list(np.array(eval_ratios[aest]['rand_first']) / eval_ratios[aest]['rand_first'][-1])

    # Print main table
    sorted_aest = sorted(aest_rows.keys())
    for aest in sorted_aest:
        for row in aest_rows[aest]:
            rows.append(row)
        for row in aest_first_choice_rows[aest]:
            rows.append(row)

    for ratio in [('own/rand', 'rand'), ('first/own', 'first'), ('first/rand_first', 'rand_first'),
                  ('first/solo own', 'first '), ('first partner\'s/solo own', 'other')]:
        nominator = ratio[0]
        denominator = ratio[1]
        for aest in sorted_aest:
            row = ['{} {} '.format(aest, nominator)] + eval_ratios[aest][denominator]
            row = [format_s % val if type(val) in [float, np.float64] else val for val in row]
            rows.append(row)

    print(tabulate(rows, headers=models))

    # Make cumulative success ratio plot
    make_success_ratio_plot(cumulative_successes, 'cumulative_success_ratios.png')
    make_success_ratio_plot(cumulative_successes,
                            'cumulative_success_ratios.pdf')


    smoothed_ratios, x = smooth_success_ratios(step_success_ratios, window=3)
    # Make non-cumulative success ratio plot
    make_success_ratio_plot(smoothed_ratios, 'success_ratios.png', x=x)
    make_success_ratio_plot(smoothed_ratios, 'success_ratios.pdf', x=x)

    # Make value and novelty solo plots
    create_solo_val_nov_plots(step_vals_novs_solo)

    # Take mean of model solo vals
    length = len(step_vals_novs_solo[list(step_vals_novs_solo.keys())[0]]['vals'])
    mean_solo_vals = np.zeros(length)
    for vals in step_vals_novs_solo.values():
        mean_solo_vals += vals['vals']
    mean_solo_vals /= len(step_vals_novs_solo)

    # Make value and novelty collab plots
    create_val_nov_plot(step_pick_stat_dict, mean_solo_vals)



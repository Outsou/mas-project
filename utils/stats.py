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


def get_step_top_values(collab_evals, pref_lists, collab_steps):
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

        for creator in creators:
            other = [agent for agent in creators if agent != creator][0]
            eval = collab_evals[collab_art][creator][0]
            val = collab_evals[collab_art][creator][1]['value']
            nov = collab_evals[collab_art][creator][1]['novelty']
            pref_list = pref_lists[creator][idx]
            aest = collab_evals[collab_art][creator][1]['aesthetic']


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

            aesthetic_top[aest]['all']['eval'].append(eval)
            aesthetic_top[aest]['all']['val'].append(val)
            aesthetic_top[aest]['all']['nov'].append(nov)
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

    aesthetic_pair_vals = {}
    top_pick_stats = {}
    aesthetic_top_stats = {}
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
        top_pick_stats = append_top_dictionaries(top_pick_stats, top_picks)
        aesthetic_top_stats = append_aest_top_dicts(aesthetic_top_stats, aesthetic_tops)

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
            for creator in creators:
                aesthetic_pair_vals[pair]['eval'].append(collab_evals[collab_art][creator][0])
                aesthetic_pair_vals[pair]['val'].append(collab_evals[collab_art][creator][1]['value'])
                aesthetic_pair_vals[pair]['nov'].append(collab_evals[collab_art][creator][1]['novelty'])
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

    top_pick_means = calculate_top_dict_means(top_pick_stats)
    aesthetic_top_pick_means = calculate_aest_top_means(aesthetic_top_stats)

    collab_eval_stats['top_pick_stats'] = top_pick_means
    collab_eval_stats['aest_top_pick_stats'] = aesthetic_top_pick_means
    collab_eval_stats['step_successes'] = step_successes / len(dirs)
    collab_eval_stats['num_of_agents'] = agents

    return collab_eval_stats


def analyze_ind_evals(dirs):
    ind_eval_stats = common_society_analysis(dirs, 'ind_evals.pkl')
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
    collab_art_stats = common_agent_analysis(dirs, pkl_name)

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
                aest_pair = tuple(sorted([aest, collab_arts_dict['caest'][i]]))
                if aest_pair not in aesthetic_pair_stats:
                    aesthetic_pair_stats[aest_pair] = {'succeeded': 0, 'failed': 0, 'rank': []}
                if collab_arts_dict['fb'][i]:
                    aesthetic_pair_stats[aest_pair]['succeeded'] += 1
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

    # Calculate aesthetic averages
    for aesthetic, caesthetics in aesthetic_stats.items():
        for caesthetic, vals in caesthetics.items():
            aesthetic_stats[aesthetic][caesthetic]['eval'] = sum(vals['eval']) / len(vals['eval'])
            aesthetic_stats[aesthetic][caesthetic]['val'] = sum(vals['val']) / len(vals['val'])
            aesthetic_stats[aesthetic][caesthetic]['nov'] = sum(vals['nov']) / len(vals['nov'])

    collab_art_stats['aesthetic_pairs'] = aesthetic_pair_stats
    collab_art_stats['aesthetic'] = aesthetic_stats

    return collab_art_stats

def analyze_own_arts(dirs):
    own_art_stats = common_agent_analysis(dirs, 'own_arts.pkl')

    first_dir = os.path.split(dirs[0])[1]
    iters = int(int(re.findall(r'i\d+', first_dir)[0][1:]) / 2)

    vals = [[] for _ in range(iters)]
    novs = [[] for _ in range(iters)]
    aest_stats = {}

    for dir in dirs:
        sub_dirs = get_dirs_in_dir(dir)
        for sub_dir in sub_dirs:
            pkl = os.path.join(sub_dir, 'own_arts.pkl')
            own_arts = pickle.load(open(pkl, 'rb'))
            general_info = pickle.load(open(os.path.join(sub_dir, 'general_info.pkl'), 'rb'))
            aest = general_info['aesthetic']

            if aest not in aest_stats:
                aest_stats[aest] = {'evals': [],
                                    'vals': [],
                                    'novs': []}

            aest_stats[aest]['evals'] += own_arts['eval']
            aest_stats[aest]['vals'] += own_arts['val']
            aest_stats[aest]['novs'] += own_arts['nov']

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

    own_art_stats['step_vals'] = vals
    own_art_stats['step_novs'] = novs
    own_art_stats['aest_means'] = aest_means
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

def make_success_ratio_plot(ratio_dict, filename):
    """Creates a success ratio plot."""
    for model, ratios in sorted(ratio_dict.items()):
        steps = len(ratios) * 2
        plt.plot(list(range(2, steps + 1, 2)), ratios, label=model)
    plt.xlabel('step')
    plt.ylabel('success ratio')
    plt.legend()
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
                    collab_eval_stats['aesthetic_pairs'][pair]['count'],
                    success_ratio,
                    collab_art_stats['aesthetic_pairs'][pair]['rank']]
        row_vals = [format_s % val if type(val) is float else val for val in row_vals]
        aesthetic_pair_rows.append(row_vals)

    print(tabulate(aesthetic_pair_rows,
                   headers=[models[-1], 'evaluation', 'value', 'novelty', 'count', 'success ratio', 'mean rank']))
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
        plt.savefig('{}_{}_top_picks.png'.format(model, val))
        plt.close()


def create_val_nov_plot(step_pick_stat_dict):
    first_key = list(step_pick_stat_dict.keys())[0]
    x = list(range(2, len(step_pick_stat_dict[first_key]['all']['val']) * 2 + 1, 2))
    for val in ['val', 'nov']:
        for model in sorted(step_pick_stat_dict.keys()):
            plt.plot(x, step_pick_stat_dict[model]['all'][val], label=model)
        plt.xlabel('step')
        plt.ylabel(val)
        plt.legend()
        plt.ylim(0.4, 0.7)
        plt.savefig('{}_collab.png'.format(val))
        plt.close()

def create_solo_val_nov_plots(step_vals_novs_solo):
    first_key = list(step_vals_novs_solo.keys())[0]
    x = list(range(2, len(step_vals_novs_solo[first_key]['vals']) * 2 + 1, 2))

    for val in ['vals', 'novs']:
        for model in sorted(step_vals_novs_solo.keys()):
            plt.plot(x, step_vals_novs_solo[model][val], label=model)
        plt.xlabel('step')
        plt.ylabel(val)
        plt.legend()
        plt.savefig('{}_solo.png'.format(val))
        plt.close()


def analyze_collab_gp_runs(path, decimals=3, exclude=None):
    """The main function to call when analyzing runs."""
    sns.set(color_codes=True)
    sns.set_context("paper")
    sns.set_style("white")
    model_dirs = sorted(get_dirs_in_dir(path))
    format_s = '%.{}f'.format(decimals)

    rows = [['Collaboration success ratio'],
            ['Success ratio conf int'],
            ['Evaluation of own collab artifacts'],
            ['Evaluation of own solo artifacts'],
            ['Value of own collab artifacts'],
            ['Value of own solo artifacts'],
            ['Novelty of own collab artifacts'],
            ['Novelty of own solo artifacts'],
            ['Overall evaluation of collab artifacts'],
            ['Overall evaluation of solo artifacts'],
            ['Overall value of collab artifacts'],
            ['Overall value of solo artifacts'],
            ['Overall novelty of collab artifacts'],
            ['Overall novelty of solo artifacts'],
            ['First choice collab evaluation'],
            ['First choice collab value'],
            ['First choice collab novelty']]
    aest_rows = {}
    aest_first_choice_rows = {}

    models = ['']
    cumulative_successes = {}
    step_success_ratios = {}
    step_pick_stat_dict = {}
    step_vals_novs_solo = {}

    for model_dir in model_dirs:
        model = os.path.split(model_dir)[1]

        if model in exclude:
            continue

        models.append(model)
        collab_eval_stats, collab_art_stats, own_art_stats, ind_eval_stats = analyze_model_dir(model_dir)

        for aest in collab_eval_stats['aest_top_pick_stats']:
            if aest not in aest_rows:
                aest_rows[aest] = [['{} own collab eval'.format(aest)],
                                   ['{} own solo eval'.format(aest)],
                                   ['{} own collab val'.format(aest)],
                                   ['{} own solo val'.format(aest)],
                                   ['{} own collab nov'.format(aest)],
                                   ['{} own solo nov'.format(aest)]]
            col_vals = [collab_eval_stats['aest_top_pick_stats'][aest]['all']['eval'],
                        own_art_stats['aest_means'][aest]['eval'],
                        collab_eval_stats['aest_top_pick_stats'][aest]['all']['val'],
                        own_art_stats['aest_means'][aest]['val'],
                        collab_eval_stats['aest_top_pick_stats'][aest]['all']['nov'],
                        own_art_stats['aest_means'][aest]['nov']]
            for i in range(len(col_vals)):
                col_vals = [format_s % val if type(val) in [float, np.float64] else val for val in col_vals]
                aest_rows[aest][i].append(col_vals[i])

            if aest not in aest_first_choice_rows:
                aest_first_choice_rows[aest] = [['{} first choice eval'.format(aest)],
                                                ['{} first choice val'.format(aest)],
                                                ['{} first choice nov'.format(aest)]]
            col_vals = [collab_eval_stats['aest_top_pick_stats'][aest]['1']['eval'],
                        collab_eval_stats['aest_top_pick_stats'][aest]['1']['val'],
                        collab_eval_stats['aest_top_pick_stats'][aest]['1']['nov']]
            for i in range(len(col_vals)):
                col_vals = [format_s % val if type(val) in [float, np.float64] else val for val in col_vals]
                aest_first_choice_rows[aest][i].append(col_vals[i])

        conf_int = collab_eval_stats['success_ratio']['conf_int']
        conf_int = (format_s % conf_int[0], format_s % conf_int[1])
        # Add column to main table
        col_vals = [collab_eval_stats['success_ratio']['mean'],
                    (conf_int),
                    collab_art_stats['avg_eval'],
                    own_art_stats['avg_eval'],
                    collab_art_stats['avg_val'],
                    own_art_stats['avg_val'],
                    collab_art_stats['avg_nov'],
                    own_art_stats['avg_nov'],
                    collab_eval_stats['avg_eval'],
                    ind_eval_stats['avg_eval'],
                    collab_eval_stats['avg_val'],
                    ind_eval_stats['avg_val'],
                    collab_eval_stats['avg_nov'],
                    ind_eval_stats['avg_nov'],
                    collab_eval_stats['top_pick_stats']['1']['eval_mean'],
                    collab_eval_stats['top_pick_stats']['1']['val_mean'],
                    collab_eval_stats['top_pick_stats']['1']['nov_mean']]

        for i in range(len(rows)):
            col_vals = [format_s % val if type(val) in [float, np.float64] else val for val in col_vals]
            rows[i].append(col_vals[i])

        # Create and print aesthetic pair table
        aesthetic_pair_rows, pairs = print_aesthetic_table(collab_eval_stats, collab_art_stats, format_s, models)
        print()

        # Create image of aesthetic pair count, success ratio and mean rank
        table = np.array(aesthetic_pair_rows)[:, -3:]
        table = np.array(table, dtype=float)
        make_table_image(np.round(table, 2), pairs, ['count', 'success ratio', 'mean rank'], '{}_aest_pairs.png'.format(model))

        # Record success ratios
        cumulative_successes[model] = get_cumulative_success_ratio(collab_eval_stats['step_successes'], collab_eval_stats['num_of_agents'] / 2)
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

    # Print main table
    for aest in sorted(aest_rows.keys()):
        for row in aest_rows[aest]:
            rows.append(row)
        for row in aest_first_choice_rows[aest]:
            rows.append(row)
    print(tabulate(rows, headers=models))

    # Make cumulative success ratio plot
    make_success_ratio_plot(cumulative_successes, 'cumulative_success_ratios.png')

    # Make non-cumulative success ratio plot
    make_success_ratio_plot(step_success_ratios, 'success_ratios.png')

    # Make value and novelty collab plots
    create_val_nov_plot(step_pick_stat_dict)

    # Make value and novelty solo plots
    create_solo_val_nov_plots(step_vals_novs_solo)

"""
Duplicated stats.py for personified stats for agents w.r.t. their skills, etc.
"""
import itertools
import operator
import os
import pickle
import numpy as np
import re
from tabulate import tabulate
import scipy.stats as st
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import table
import pandas as pd

from experiments.collab.chk_runs import get_last_lines
from utils.util import primitives


# Skills shared by all agents
SHARED_SKILLS = ['mul', 'safe_div', 'add', 'sub', 'safe_mod']
SKILLS = list(primitives.keys())


def ci(std, n, conf=99):
    if conf == 99:
        z = 2.576
    elif conf == 98:
        z = 2.326
    elif conf == 95:
        z = 1.96
    elif conf == 90:
        z = 1.645
    else:
        raise Exception("Confidence interval not recognized!")

    plus_minus = z * (std / np.sqrt(n))
    return plus_minus


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

        for creator in creators:
            other = [agent for agent in creators if agent != creator][0]

            if other_vals:
                val_agent = other
            else:
                val_agent = creator

            eval = collab_evals[collab_art][val_agent][0]
            val = collab_evals[collab_art][val_agent][1]['value']
            nov = collab_evals[collab_art][val_agent][1]['novelty']

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


def calculate_collab_partner_means(all_collab_partners):
    sums = []
    n_agents = len(all_collab_partners)
    for l in all_collab_partners:
        for i in range(len(l)):
            if len(sums) == i:
                sums.append(l[i])
            else:
                sums[i] += l[i]
    return [s / n_agents for s in sums]


def create_collab_partner_plot(means, means_rev):
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    fig, ax1 = plt.subplots()
    ax1.set_title("Cumulative collaboration partner count for schemes")
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Cumulative number of collaboration partners')
    i = 0
    models = sorted(list(means.items()), key=operator.itemgetter(0))
    from utils.plot_styling import LINE_STYLES
    for model, mean in models:
        x = list(range(2, 202, 2))
        ax1.plot(x, mean, LINE_STYLES[i], label=model)
        i += 1

    ax2 = ax1.twinx()
    ax2.set_ylabel("Number of collaboration partners in iterations left")
    models = sorted(list(means_rev.items()), key=operator.itemgetter(0))
    i = 0
    for model, mean in models:
        x = list(range(2, 202, 2))
        ax2.plot(x, mean, LINE_STYLES[i], label=model)
        i += 1
    ax1.legend(loc='lower center')
    fig.savefig("cumulative_collaboration_partners.pdf")


def analyze_model_dir(path):
    # Get directories of valid runs
    dirs = []
    lines = get_last_lines(path)
    for line in lines:
        if 'Run finished' in line:
            dirs.append(line.split(': ')[0])

    #collab_eval_stats = analyze_collab_evals(dirs)
    #collab_art_stats = analyze_collab_arts(dirs)
    #own_art_stats, agent_info = analyze_own_arts(dirs)
    #ind_eval_stats = analyze_ind_evals(dirs)
    cumulative_collab_partner_means, cumulative_collab_partners_rev_means = analyze_agent_collab_skills(dirs, path)

    return cumulative_collab_partner_means, cumulative_collab_partners_rev_means

def get_agent_infos(dirs):
    # key:dir value: dict
    # inner dict key: agent addr, value: dict (aesthetic and skills as keys)
    agent_info = {}

    for dir in dirs:
        agent_info[dir] = {}
        sub_dirs = get_dirs_in_dir(dir)
        for sub_dir in sub_dirs:
            gi = pickle.load(open(os.path.join(sub_dir, 'general_info.pkl'), 'rb'))
            agent_info[dir][gi['addr']] = {'aesthetic': gi['aesthetic'],
                                           'skills': gi['pset_names']}
    return agent_info


def analyze_agent_collab_skills(dirs, path):
    """Analyze the effect of agent skills on the collaboration results.
    """
    pkl_name = 'collab_arts.pkl'
    agent_info = get_agent_infos(dirs)
    analyze_agent_info(agent_info)
    var_funcs = ['perlin1', 'perlin2', 'simplex2', 'plasma']
    various_fb = []
    nonvar_fb = []
    both_various_fb = []
    init_various_fb = []
    other_various_fb = []
    none_various_fb = []
    other_nonvar_fb = []
    n_collab_partners_all = []
    n_collab_partners_rev_all = []

    for dir in dirs:
        agents = agent_info[dir]
        pref_lists = pickle.load(open(os.path.join(dir, 'pref_lists.pkl'), 'rb'))
        sub_dirs = get_dirs_in_dir(dir)
        for sub_dir in sub_dirs:
            with open(os.path.join(sub_dir, pkl_name), 'rb') as pkl:
                collab_arts_dict = pickle.load(pkl)
            gi = pickle.load(
                open(os.path.join(sub_dir, 'general_info.pkl'), 'rb'))
            agent = agents[gi['addr']]
            pref_list = pref_lists[gi['addr']]

            # Number of overall collaboration partners per collaboration iter.
            n_collab_partners = []
            # Previous collaboration partners
            prev_collab_partners = []
            n_collab_partners_rev = [0 for _ in range(len(collab_arts_dict['fb']))]
            prev_collab_partners_rev = []
            same_skills_succ = []   # Same skills for succeeded collab
            same_skills_fail = []   # Same skills for failed collab
            same_skills_all = []    # Same skills for all collab
            same_skills_init = []   # Same skills for init collab
            same_skills_first = []
            all_collab = []
            var_collab = []
            var_collab_fb = []
            has_various = any([e in agent['skills'] for e in var_funcs])
            if has_various:
                various_fb.append(collab_arts_dict['fb'])
            else:
                nonvar_fb.append(collab_arts_dict['fb'])

            for i in reversed(range(len(collab_arts_dict['fb']))):
                caddr = collab_arts_dict['caddr'][i]
                if caddr in prev_collab_partners_rev:
                    n_collab_partners_rev[i] = n_collab_partners_rev[i + 1]
                else:
                    if len(prev_collab_partners_rev) == 0:
                        n_collab_partners_rev[i] = 1
                        prev_collab_partners_rev.append(caddr)
                    else:
                        n = n_collab_partners_rev[i + 1] + 1
                        n_collab_partners_rev[i] = n
                        prev_collab_partners_rev.append(caddr)

            fb = 0  # Collab successes until now
            for i in range(len(collab_arts_dict['fb'])):
                caddr = collab_arts_dict['caddr'][i]
                if caddr in prev_collab_partners:
                    n = n_collab_partners[-1]
                    n_collab_partners.append(n)
                else:
                    if len(n_collab_partners) == 0:
                        n_collab_partners.append(1)
                        prev_collab_partners.append(caddr)
                    else:
                        n = n_collab_partners[-1] + 1
                        n_collab_partners.append(n)
                        prev_collab_partners.append(caddr)
                #print(caddr)
                same_skills = agent[caddr]['intersection']
                unique_skills = agent[caddr]['unique']
                same_skills_all.append(len(same_skills))
                other_has_various = any(
                    [e in agents[caddr]['skills'] for e in var_funcs])
                if caddr == pref_list[i][0] and collab_arts_dict['cinit'][i]:
                    same_skills_first.append(len(same_skills))
                if collab_arts_dict['cinit'][i]:
                    same_skills_init.append(len(same_skills))
                if collab_arts_dict['fb'][i]:
                    same_skills_succ.append(len(same_skills))
                    all_collab.append(collab_arts_dict['val'][fb])
                    if has_various or other_has_various:
                        var_collab.append(collab_arts_dict['val'][fb])
                    if collab_arts_dict['cinit'][i]:
                        if has_various and other_has_various:
                            both_various_fb.append(1)
                        elif other_has_various:
                            other_various_fb.append(1)
                        elif has_various:
                            init_various_fb.append(1)
                        else:
                            none_various_fb.append(1)

                    fb += 1
                else:
                    same_skills_fail.append(len(same_skills))
                    if collab_arts_dict['cinit'][i]:
                        if has_various and other_has_various:
                            both_various_fb.append(0)
                        elif other_has_various:
                            other_various_fb.append(0)
                        elif has_various:
                            init_various_fb.append(0)
                        else:
                            none_various_fb.append(0)

            n_collab_partners_rev_all.append(n_collab_partners_rev)
            n_collab_partners_all.append(n_collab_partners)
            agent['same_skills_succ'] = same_skills_succ
            agent['same_skills_fail'] = same_skills_fail
            agent['same_skills_all'] = same_skills_all
            agent['same_skills_init'] = same_skills_init
            agent['same_skills_first'] = same_skills_first
            agent['same_skills_succ_avg'] = np.mean(same_skills_succ)
            if len(same_skills_fail) > 0:
                agent['same_skills_fail_avg'] = np.mean(same_skills_fail)
            else:
                agent['same_skills_fail_avg'] = 0.0
            agent['same_skills_all_avg'] = np.mean(same_skills_all)
            agent['same_skills_init_avg'] = np.mean(same_skills_init)
            if len(same_skills_first) > 0:
                agent['same_skills_first_avg'] = np.mean(same_skills_first)
            else:
                agent['same_skills_first_avg'] = 0.0

            agent['all_collab_val'] = all_collab
            agent['various_collab_val'] = var_collab
            agent['all_collab_val_avg'] = np.mean(all_collab)
            agent['various_collab_val_avg'] = np.mean(var_collab)

    print_agent_info_statistics(agent_info, both_various_fb, init_various_fb,
                                other_various_fb, none_various_fb)

    mean_collab_partners = calculate_collab_partner_means(n_collab_partners_all)
    mean_collab_partners_rev = calculate_collab_partner_means(n_collab_partners_rev_all)
    return mean_collab_partners, mean_collab_partners_rev


def analyze_agent_info(agent_info):
    # Agents have on average 2.66 same skills
    means = []
    aest_means = []

    for dir, agents in agent_info.items():
        for addr in agents:
            skills = set(agents[addr]['skills'])
            aest = agents[addr]['aesthetic']
            for addr2 in agents:
                if addr != addr2:
                    skills2 = set(agents[addr2]['skills'])
                    aest2 = agents[addr2]['aesthetic']
                    same_skills = skills & skills2
                    diff1 = skills - skills2
                    diff2 = skills2 - skills
                    agents[addr][addr2] = {'intersection': same_skills,
                                           'unique': diff1}
                    agents[addr2][addr] = {'intersection': same_skills,
                                           'unique': diff2}
                    #print(same_skills)
                    means.append(len(same_skills))
                    if aest == aest2:
                        aest_means.append(len(same_skills))

    mean_all = np.mean(means)
    mean_aest = np.mean(aest_means)
    print("AVG same skills: {:.3f} (std: {:.3f})".format(mean_all, np.std(means)))
    print("AVG same skills within aesthetic: {:.3f} (std: {:.3f})".format(mean_aest, np.std(aest_means)))


def print_agent_info_statistics(agent_info, both_various_fb, init_various_fb,
                                other_various_fb, none_various_fb):

    def get_all_info(agents, key, filter=None):
        if filter is not None:
            agents = [a for a in agents if a[filter[0]] == filter[1]]
        key_values = list(itertools.chain(*[a[key] for a in agents]))
        mean = np.mean(key_values)
        std = np.std(key_values)
        conf_interval = ci(std, len(key_values))
        return mean, std, conf_interval

    def get_list_measures(collab_list):
        clist = collab_list
        mean = np.mean(clist)
        std = np.std (clist)
        conf_interval = ci(std, len(clist))
        return mean, std, conf_interval

    all_agents = [list(agents.values()) for agents in agent_info.values()]
    all_agents = list(itertools.chain(*all_agents))
    same_succs = get_all_info(all_agents, 'same_skills_succ')
    same_fails = get_all_info(all_agents, 'same_skills_fail')
    same_alls = get_all_info(all_agents, 'same_skills_all')
    same_firsts = get_all_info(all_agents, 'same_skills_first')

    print("Mean same skills for succeeded collab: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(
        *same_succs))
    print("Mean same skills for failed collab: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(
        *same_fails))
    print("Mean same skills for all collab: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(
        *same_alls))
    print("Mean same skills for first choice collab: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(
        *same_firsts))

    both_vars = get_list_measures(both_various_fb)
    init_vars = get_list_measures(init_various_fb)
    other_vars = get_list_measures(other_various_fb)
    none_vars = get_list_measures(none_various_fb)
    print("Mean collab success with both various funcs: {:.3f} (std: {:.3f}) CI99% +-{:.3f} (n={})".format(
        *both_vars, len(both_various_fb)))
    print("Mean collab success with init various funcs: {:.3f} (std: {:.3f}) CI99% +-{:.3f} (n={})".format(
        *init_vars, len(init_various_fb)))
    print("Mean collab success with other various funcs: {:.3f} (std: {:.3f}) CI99% +-{:.3f} (n={})".format(
        *other_vars, len(other_various_fb)))
    print("Mean collab success with none various funcs: {:.3f} (std: {:.3f}) CI99% +-{:.3f} (n={})".format(
        *none_vars, len(none_various_fb)))


    all_collab_vals = get_all_info(all_agents, key='all_collab_val')
    all_collab_vals_gcf = get_all_info(all_agents, key='all_collab_val', filter=('aesthetic', 'global_contrast_factor'))
    all_collab_vals_bl = get_all_info(all_agents, key='all_collab_val', filter=('aesthetic', 'benford'))
    all_collab_vals_ent = get_all_info(all_agents, key='all_collab_val', filter=('aesthetic', 'entropy'))
    all_collab_vals_symm = get_all_info(all_agents, key='all_collab_val', filter=('aesthetic', 'symm'))
    all_collab_vals_fda = get_all_info(all_agents, key='all_collab_val', filter=('aesthetic', 'fd_aesthetics'))

    var_collab_vals = get_all_info(all_agents, key='various_collab_val')
    var_collab_vals_gcf = get_all_info(all_agents, key='various_collab_val', filter=('aesthetic', 'global_contrast_factor'))
    var_collab_vals_bl = get_all_info(all_agents, key='various_collab_val', filter=('aesthetic', 'benford'))
    var_collab_vals_ent = get_all_info(all_agents, key='various_collab_val', filter=('aesthetic', 'entropy'))
    var_collab_vals_symm = get_all_info(all_agents, key='various_collab_val', filter=('aesthetic', 'symm'))
    var_collab_vals_fda = get_all_info(all_agents, key='various_collab_val', filter=('aesthetic', 'fd_aesthetics'))

    print("All collab val avg: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*all_collab_vals))
    print("Various collab val avg: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*var_collab_vals))
    print("All collab val avg gfc: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*all_collab_vals_gcf))
    print("Various collab val avg gcf: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*var_collab_vals_gcf))
    print("All collab val avg bl: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*all_collab_vals_bl))
    print("Various collab val avg bl: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*var_collab_vals_bl))
    print("All collab val avg ent: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*all_collab_vals_ent))
    print("Various collab val avg ent: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*var_collab_vals_ent))
    print("All collab val avg symm: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*all_collab_vals_symm))
    print("Various collab val avg symm: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*var_collab_vals_symm))
    print("All collab val avg fda: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*all_collab_vals_fda))
    print("Various collab val avg fda: {:.3f} (std: {:.3f}) CI99% +-{:.3f}".format(*var_collab_vals_fda))


def analyze_collab_gp_runs(path, decimals=3, exclude=[]):
    """The main function to call when analyzing runs."""
    sns.set(color_codes=True)
    sns.set_context("paper")
    sns.set_style("white")
    model_dirs = sorted(get_dirs_in_dir(path))
    format_s = '%.{}f'.format(decimals)
    models = ['']
    means = {}
    means_rev = {}

    for model_dir in model_dirs:
        model = os.path.split(model_dir)[1]
        print("Model: {}\n***************************************************".format(model))

        if model in exclude:
            continue

        models.append(model)
        cumulative_collab_partner_means, cumulative_collab_partners_rev_means = analyze_model_dir(model_dir)
        means[model] = cumulative_collab_partner_means
        means_rev[model] = cumulative_collab_partners_rev_means
        print("***************************************************\n")

    create_collab_partner_plot(means, means_rev)





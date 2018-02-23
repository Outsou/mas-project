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
from utils.plot_styling import *


# Skills shared by all agents
SHARED_SKILLS = ['mul', 'safe_div', 'add', 'sub', 'safe_mod']
SKILLS = list(primitives.keys())

skill_regexs = {s: re.compile(s) for s in SKILLS}


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


def get_dirs_in_dir(path):
    dirs = [os.path.join(path, file) for file in os.listdir(path) if os.path.isdir(os.path.join(path, file))]
    return dirs


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


def create_collab_partner_plot(means, means_rev, separated=True):
    fig_size = BASE_FIG_SIZE
    sns.set()
    sns.set_style("white")
    sns.set_context("paper")
    fig, ax1 = plt.subplots(figsize=fig_size)
    #ax1.set_title("Cumulative collaboration partner count for schemes")
    ax1.set_xlabel('Iteration')
    ax1.set_ylabel('Distinct partners (cumulative)')
    i = 0
    # Means is a dictionary with (internal) model names as keys and values are
    # vectors of numbers.
    smeans = sorted(means.items(), key=lambda x: MODEL_ORDER.index(x[0]))
    for model, mean in smeans:
        style = MODEL_STYLES[model]
        x = list(range(2, 202, 2))
        ax1.plot(x, mean, style['line style'], dashes=style['dashes'], label=style['label'], color=style['color'])
        i += 1

    if separated:
        ax1.legend(loc='lower right')
        plt.tight_layout()
        fig.savefig("cumulative_collaboration_partners.pdf")
        fig, ax2 = plt.subplots(figsize=fig_size)
        ax2.set_xlabel("Iteration")
    else:
        ax2 = ax1.twinx()
    ax2.set_ylabel("Distinct partners (remaining)")
    i = 0
    smeans = sorted(means_rev.items(), key=lambda x: MODEL_ORDER.index(x[0]))
    for model, mean in smeans:
        style = MODEL_STYLES[model]
        x = list(range(2, 202, 2))
        ax2.plot(x, mean, style['line style'], dashes=style['dashes'], label=style['label'], color=style['color'])
        i += 1
    ax2.legend(loc='lower left')
    plt.tight_layout()
    fig.savefig("remaining_collaboration_partners.pdf")


def analyze_model_dir(path, old_collab):
    # Get directories of valid runs
    dirs = []
    lines = get_last_lines(path)
    for line in lines:
        if 'Run finished' in line:
            dirs.append(line.split(': ')[0])
    agent_info = get_agent_infos(dirs)
    analyze_agent_info(agent_info)
    #analyze_artifact_skills(dirs, path, agent_info)
    rets = analyze_agent_collab_skills(dirs, path, agent_info, old_collab)
    return rets


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


def update_func_fbs(all_funcs, a1, a2, fb):
    for fname in all_funcs.keys():
        init_has = has_func(a1, fname)
        other_has = has_func(a2, fname)
        if init_has and other_has:
            all_funcs[fname]['both_func_fb'].append(fb)
        elif init_has:
            all_funcs[fname]['init_func_fb'].append(fb)
        elif other_has:
            all_funcs[fname]['other_func_fb'].append(fb)
        else:
            all_funcs[fname]['none_func_fb'].append(fb)


def has_funcs(agent, func_names):
    return {fname: has_func(agent, fname) for fname in func_names}


def has_func(agent, func_name):
    return func_name in agent['skills']


def get_artifact_funcs(func_str):
    s = []
    count = 0
    for skill, regex in skill_regexs.items():
        if regex.search(func_str) is not None:
            s.append(skill)
            ret = regex.findall(func_str)
            count += len(ret)

    return s, count


def get_all_artifact_funcs(sub_dir, agent_skills, agents):
    own_funcs = []
    collab_funcs = []

    fls = [e for e in os.listdir(sub_dir) if os.path.isfile(os.path.join(sub_dir, e))]
    for fl in fls:
        if fl[-3:] == 'txt':
            with open(os.path.join(sub_dir, fl), 'r') as f:
                func_str = f.read().strip()
                if len(fl.split("-")) == 2:
                    c = fl.split("-")[1]
                    caddr = "{}://{}:{}/{}".format(*c.split("_")[:4])
                    oskills = agents[caddr]['skills']
                    #print(caddr, oskills)
                    ret = analyze_artifact_func(func_str, agent_skills, oskills)
                    collab_funcs.append(ret)
                else:
                    ret = analyze_artifact_func(func_str, None, None)
                    own_funcs.append(ret[:2])
    return own_funcs, collab_funcs


def analyze_artifact_func(func, agent_skills=None, other_skills=None):
    askills = None
    oskills = None
    s, count = get_artifact_funcs(func)
    # print(len(s), s)
    if agent_skills is not None:
        sg = []
        for g in agent_skills:
            if g in s:
                sg.append(g)
        askills = sg
    if other_skills is not None:
        og = []
        for g in other_skills:
            if g in s:
                og.append(g)
        oskills = og
    return s, count, askills, oskills


def get_agent_dicts(sub_dir):
    collab_pkl = 'collab_arts.pkl'
    own_pkl = 'own_arts.pkl'
    gi_pkl = 'general_info.pkl'
    with open(os.path.join(sub_dir, collab_pkl), 'rb') as pkl:
        collab_arts_dict = pickle.load(pkl)
    with open(os.path.join(sub_dir, own_pkl), 'rb') as pkl:
        own_arts_dict = pickle.load(pkl)
    with open(os.path.join(sub_dir, gi_pkl), 'rb') as pkl:
        general_info_dict = pickle.load(pkl)
    return collab_arts_dict, own_arts_dict, general_info_dict


def analyze_artifact_skills(dirs, path, agent_info):

    def get_mean_std(l):
        return np.mean(l), np.std(l)

    all_funcs = list(primitives.keys())
    all_own_skills = []
    all_own_skill_counts = []
    all_col_skills = []
    all_col_skill_counts = []
    all_agent_skills = []
    all_other_skills = []

    for dir in dirs:
        agents = agent_info[dir]
        pref_lists = pickle.load(open(os.path.join(dir, 'pref_lists.pkl'), 'rb'))
        sub_dirs = get_dirs_in_dir(dir)
        for sub_dir in sub_dirs:
            collab_arts_dict, own_arts_dict, gi = get_agent_dicts(sub_dir)
            agent = agents[gi['addr']]
            pref_list = pref_lists[gi['addr']]
            own_funcs, collab_funcs = get_all_artifact_funcs(sub_dir, agent['skills'], agents)
            own_skills, own_skill_counts = zip(*own_funcs)
            #print(len(agent['skills']), agent['skills'])
            col_skills, col_skill_counts, agent_skills, other_skills = zip(*collab_funcs)
            all_own_skills.extend(own_skills)
            all_own_skill_counts.extend(own_skill_counts)
            all_col_skills.extend(col_skills)
            all_col_skill_counts.extend(col_skill_counts)
            all_agent_skills.extend(agent_skills)
            all_other_skills.extend(other_skills)

    mos, sos = get_mean_std([len(s) for s in all_own_skills])
    mosc, sosc = get_mean_std([s for s in all_own_skill_counts])
    mcs, scs = get_mean_std([len(s) for s in all_col_skills])
    mcsc, scsc = get_mean_std([s for s in all_col_skill_counts])
    mas, sas = get_mean_std([len(s) for s in all_agent_skills])
    maos, saos = get_mean_std([len(s) for s in all_other_skills])
    print("Mean own skills: {:.3f}, std: {:.3f}  ({:.3f})".format(mos, sos, mosc))
    print("Mean col skills: {:.3f} std: {:.3f} ({:.3f})".format(mcs, scs, mcsc))
    print("Mean agent skills in col: {:.3f} std: {:.3f}".format(mas, sas))
    print("Mean other skills in col: {:.3f} std: {:.3f}".format(maos, saos))


def analyze_agent_collab_skills(dirs, path, agent_info, old_collab=True):
    """Analyze the effect of agent skills on the collaboration results.
    """
    pkl_name = 'collab_arts.pkl'
    var_funcs = ['perlin1', 'perlin2', 'simplex2', 'plasma']
    all_funcs = {e: {'both_func_fb': [], 'init_func_fb': [], 'other_func_fb': [], 'none_func_fb': []} for e in list(primitives.keys())}
    both_various_fb = []
    init_various_fb = []
    other_various_fb = []
    none_various_fb = []
    n_collab_partners_all = []
    n_collab_partners_rev_all = []
    print(len(dirs), path)
    if old_collab:
        all_collab_selections = [[0 for _ in range(5)] for _ in range(5)]
        aest2idx = {'benford': 0, 'entropy': 1, 'global_contrast_factor': 2, 'symm': 3, 'fd_aesthetics': 4}
    else:
        all_collab_selections = [[0 for _ in range(2)] for _ in range(2)]
        aest2idx = {'entropy': 0, 'complexity': 1}

    for dir in dirs:
        #collab_selections = [[0 for _ in range(5)] for _ in range(5)]
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
            has_various = any([has_func(agent, e) for e in var_funcs])

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
                agent2 = agents[caddr]

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
                other_has_various = any([has_func(agent2, e) for e in var_funcs])
                #print([has_func(agent, e) for e in var_funcs], [has_func(agent2, e) for e in var_funcs])
                if caddr == pref_list[i][0] and collab_arts_dict['cinit'][i]:
                    same_skills_first.append(len(same_skills))
                if collab_arts_dict['cinit'][i]:
                    idx1 = aest2idx[agent['aesthetic']]
                    idx2 = aest2idx[agent2['aesthetic']]
                    all_collab_selections[idx1][idx2] += 1
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
                        update_func_fbs(all_funcs, agent, agent2, 1)

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
                        update_func_fbs(all_funcs, agent, agent2, 0)

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
        #all_collab_selections.append(collab_selections)

    acs = np.array(all_collab_selections)
    acs = acs / len(dirs)
    #print(acs)
    mean_collab_selections = acs

    if old_collab:
        for i,e in enumerate(acs):
            print("{}: {} {} {:.3f}".format(i, AEST_ORDER[i], ["{:.3f}".format(k) for k in e], sum(e)))
    else:
        for i,e in enumerate(acs):
            print("{}: {} {} {:.3f}".format(i, AEST_ORDER_NEW[i], ["{:.3f}".format(k) for k in e], sum(e)))


    print_agent_info_statistics(agent_info, both_various_fb, init_various_fb,
                                other_various_fb, none_various_fb, all_funcs)

    mean_collab_partners = calculate_collab_partner_means(n_collab_partners_all)
    mean_collab_partners_rev = calculate_collab_partner_means(n_collab_partners_rev_all)
    return mean_collab_partners, mean_collab_partners_rev, mean_collab_selections


def compute_mean_selections(all_collab_selections):
    means = [[0 for _ in range(5)] for _ in range(5)]
    for cs in all_collab_selections:
        for i in range(len(cs)):
            for j in range(len(cs[0])):
                means[i][j] += cs[i][j]

    for i in range(len(means)):
        for j in range(len(means[0])):
            means[i][j] /= len(all_collab_selections)

    return means


def create_aesthetic_collab_heatmap(collab_selections, old_collab=True):

    if old_collab:
        return create_aesthetic_collab_heatmap_old(collab_selections)
    else:
        return create_aesthetic_collab_heatmap_new(collab_selections)


def create_aesthetic_collab_heatmap_new(collab_selections):
    sns.set_style("white")
    sns.set_context("paper")
    from matplotlib.colors import LogNorm

    for model, c_selections in collab_selections.items():
        fig, ax = plt.subplots(figsize=(3, 3))
        cs = np.asarray(c_selections)
        ax = sns.heatmap(cs,
                         xticklabels=AEST_SHORT_LABELS_NEW,
                         yticklabels=AEST_SHORT_LABELS_NEW,
                         cmap="Greys", vmin=0, vmax=1000,
                         square=True,
                         cbar=True)
        #plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("{}_aest_collabs_heatmap.pdf".format(model))
        plt.close()

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.5, 2.5))
    cs1 = np.asarray(collab_selections['Q1'])
    cs2 = np.asarray(collab_selections['Q2'])
    cs3 = np.asarray(collab_selections['Q3'])
    ax = sns.heatmap(cs1, ax=ax1,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=160,
                     square=True,
                     cbar=False)
    ax = sns.heatmap(cs2, ax=ax2,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=160,
                     square=True,
                     cbar=False)
    ax = sns.heatmap(cs3, ax=ax3,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=160,
                     square=True,
                     cbar=True)

    plt.tight_layout()
    plt.savefig("Q123_aest_collabs_heatmap.pdf".format(model))
    plt.close()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10.5, 2.3))
    cs1 = np.asarray(collab_selections['random'])
    cs2 = np.asarray(collab_selections['hedonic-Q'])
    cs3 = np.asarray(collab_selections['state-Q'])
    cs4 = np.asarray(collab_selections['state-Q2'])
    ax = sns.heatmap(cs1, ax=ax1,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=150.5,
                     square=True,
                     cbar=False)
    ax = sns.heatmap(cs2, ax=ax2,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=150.5,
                     square=True,
                     cbar=False)
    ax = sns.heatmap(cs3, ax=ax3,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=150.5,
                     square=True,
                     cbar=False)
    ax = sns.heatmap(cs4, ax=ax4,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=150.5,
                     square=True,
                     cbar=True)

    plt.tight_layout()
    plt.savefig("all_aest_collabs_heatmap_new.pdf".format(model))
    plt.close()
    """


def create_aesthetic_collab_heatmap_old(collab_selections):
    sns.set_style("white")
    sns.set_context("paper")
    from matplotlib.colors import LogNorm

    for model, c_selections in collab_selections.items():
        fig, ax = plt.subplots(figsize=(3, 3))
        cs = np.asarray(c_selections)
        ax = sns.heatmap(cs,
                         xticklabels=AEST_SHORT_LABELS,
                         yticklabels=AEST_SHORT_LABELS,
                         cmap="Greys", vmin=0, vmax=160,
                         square=True,
                         cbar=True)
        #plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig("{}_aest_collabs_heatmap.pdf".format(model))
        plt.close()

    """
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(8.5, 2.5))
    cs1 = np.asarray(collab_selections['Q1'])
    cs2 = np.asarray(collab_selections['Q2'])
    cs3 = np.asarray(collab_selections['Q3'])
    ax = sns.heatmap(cs1, ax=ax1,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=160,
                     square=True,
                     cbar=False)
    ax = sns.heatmap(cs2, ax=ax2,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=160,
                     square=True,
                     cbar=False)
    ax = sns.heatmap(cs3, ax=ax3,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=160,
                     square=True,
                     cbar=True)

    plt.tight_layout()
    plt.savefig("Q123_aest_collabs_heatmap.pdf".format(model))
    plt.close()

    fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(10.5, 2.3))
    cs1 = np.asarray(collab_selections['Q1'])
    cs2 = np.asarray(collab_selections['Q2'])
    cs3 = np.asarray(collab_selections['Q3'])
    cs4 = np.asarray(collab_selections['lr'])
    ax = sns.heatmap(cs1, ax=ax1,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=150.5,
                     square=True,
                     cbar=False)
    ax = sns.heatmap(cs2, ax=ax2,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=150.5,
                     square=True,
                     cbar=False)
    ax = sns.heatmap(cs3, ax=ax3,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=150.5,
                     square=True,
                     cbar=False)
    ax = sns.heatmap(cs4, ax=ax4,
                     xticklabels=AEST_SHORT_LABELS,
                     yticklabels=AEST_SHORT_LABELS,
                     cmap="Greys", vmin=0, vmax=150.5,
                     square=True,
                     cbar=True)

    plt.tight_layout()
    plt.savefig("all_aest_collabs_heatmap.pdf".format(model))
    plt.close()
    """


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
                                other_various_fb, none_various_fb, all_funcs):

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

    def print_func(all_funcs, fname):
        both_vars = get_list_measures(all_funcs[fname]['both_func_fb'])
        init_vars = get_list_measures(all_funcs[fname]['init_func_fb'])
        other_vars = get_list_measures(all_funcs[fname]['other_func_fb'])
        none_vars = get_list_measures(all_funcs[fname]['none_func_fb'])
        print(
            "Mean collab success with both {}: {:.3f} (std: {:.3f}) CI99% +-{:.3f} (n={})".format(
                fname, *both_vars, len(all_funcs[fname]['both_func_fb'])))
        print(
            "Mean collab success with init {}: {:.3f} (std: {:.3f}) CI99% +-{:.3f} (n={})".format(
                fname, *init_vars, len(all_funcs[fname]['init_func_fb'])))
        print(
            "Mean collab success with other {}: {:.3f} (std: {:.3f}) CI99% +-{:.3f} (n={})".format(
                fname, *other_vars, len(all_funcs[fname]['other_func_fb'])))
        print(
            "Mean collab success with none {}: {:.3f} (std: {:.3f}) CI99% +-{:.3f} (n={})".format(
                fname, *none_vars, len(all_funcs[fname]['none_func_fb'])))


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

    for fname in all_funcs.keys():
        print_func(all_funcs, fname)

    '''
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
    '''


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
    c_selections = {}
    old_collab = True
    for m in model_dirs:
        if m.endswith('state-Q'):
            old_collab = False

    for model_dir in model_dirs:
        model = os.path.split(model_dir)[1]
        print("Model: {}\n***************************************************".format(model))

        if model in exclude:
            continue

        models.append(model)
        cumulative_collab_partner_means, cumulative_collab_partners_rev_means, collab_selections = analyze_model_dir(model_dir, old_collab)
        means[model] = cumulative_collab_partner_means
        means_rev[model] = cumulative_collab_partners_rev_means
        c_selections[model] = collab_selections
        print(np.min(collab_selections), np.max(collab_selections))
        print("***************************************************\n")

    create_aesthetic_collab_heatmap(c_selections, old_collab)
    create_collab_partner_plot(means, means_rev)





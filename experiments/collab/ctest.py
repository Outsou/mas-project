"""Command line script to run collaboration tests.
"""

import logging
import pprint
import random
import copy
import socket
import argparse
import os

import aiomas
import numpy as np
from creamas import Simulation

from agents import GPImageAgent
from experiments.collab import collab_exp as coe
from experiments.collab.base import CollabSimulation


HOST = socket.gethostname()


def get_run_id(path):
    path = path if len(path) > 0 else "."
    d = os.listdir(path)
    run_id = 1
    for e in d:
        if os.path.isdir(os.path.join(path, e)):
            run_id += 1
    return run_id


def run_sim(params, save_path, log_folder):
    nslaves = 8
    num_of_steps = params['num_of_steps']
    pop_size = params['population_size']
    shape = params['shape']
    sample_size = params['pset_sample_size']
    menv = coe.create_environment(num_of_slaves=nslaves, save_folder=save_path)
    r = coe.create_agents('experiments.collab.base:GPCollaborationAgent',
                          menv, params, log_folder, save_path, pop_size, shape,
                          sample_size)
    coe.create_agent_connections(menv, params['agents'])

    sim = CollabSimulation(menv,
                           precallback=menv.match_collab_partners,
                           callback=menv.post_cbk,
                           log_folder=log_folder)

    # RUN SIMULATION
    for i in range(num_of_steps):
        sim.async_step()

    rets = menv.save_artifact_info()
    sim.end()
    return rets

if __name__ == "__main__":
    # Command line argument parsing
    desc = "Command line script to run collaboration test runs."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-a', metavar='agents', type=int, dest='agents',
                        help="Number of agents.", default=20)
    parser.add_argument('-s', metavar='steps', type=int, dest='steps',
                        help="Number of simulation steps.", default=200)
    parser.add_argument('-m', metavar='model', type=str, dest='model',
                        help="Learning model to be used.", default='random')
    parser.add_argument('-n', metavar='novelty', type=int, dest='novelty',
                        help="Novelty weight.", default=0.2)
    parser.add_argument('-l', metavar='folder', type=str, dest='save_folder',
                        help="Base folder to save test run. The actual save folder is created as a subfolder to the base folder.",
                        default="runs")
    parser.add_argument('-r', metavar='run ID', type=int, dest='run_id',
                        help="Run ID, if needed to set manually.", required=False)

    args = parser.parse_args()

    # DEFINE TEST PARAMETERS
    params = coe.DEFAULT_PARAMS
    params['agents'] = args.agents
    params['novelty_weight'] = args.novelty
    params['num_of_steps'] = args.steps
    learning_model = args.model
    base_path = os.path.join(".", args.save_folder)
    os.makedirs(base_path, exist_ok=True)
    log_folder = 'foo'
    run_id = args.run_id if args.run_id is not None else get_run_id(base_path)

    # CREATE SIMULATION AND RUN
    run_folder = 'r{:0>4}m{}a{}e{}i{}_{}'.format(
        run_id, learning_model, params['agents'], len(params['aesthetic_list']),
        params['num_of_steps'], HOST)
    if len(base_path) > 0:
        run_folder = os.path.join(base_path, run_folder)
        log_folder = run_folder

    # Error if the run folder exists for some reason. Should not happen if no
    # additional folders are spawned (or folders
    os.makedirs(run_folder, exist_ok=False)
    print("Initializing run with {} agents, {} aesthetic measures, {} model, "
        "{} steps.".format(args.agents, len(params['aesthetic_list']),
                           args.model, args.steps))
    print("Saving run output to {}".format(run_folder))
    #os.makedirs(log_folder, exist_ok=True)
    run_sim(params, run_folder, log_folder)


import logging
import pprint
import random
import copy

import aiomas
import numpy as np
from creamas import Simulation

from agents import GPImageAgent
from experiments.collab import collab_exp as coe
from experiments.collab.base import CollabSimulation


if __name__ == "__main__":
    # DEFINE TEST PARAMETERS
    num_of_steps = 8
    pop_size = 20
    shape = (64, 64)
    sample_size = 8
    #create_kwargs = coe.get_create_kwargs(20, (64, 64))
    menv = coe.create_environment(num_of_slaves=4)
    params = coe.DEFAULT_PARAMS
    params['agents'] = 4
    params['novelty_weight'] = 0.2

    # END PARAM DEF

    # CREATE SIMULATION AND RUN
    path = 'collab_test_a{}_e{}_i{}'\
        .format(params['agents'], len(params['aesthetic_list']), num_of_steps)
    log_folder = 'foo'
    coe._init_data_folder(path)

    r = coe.create_agents('experiments.collab.base:GPCollaborationAgent',
                          menv, params, log_folder, path, pop_size, shape,
                          sample_size)
    coe.create_agent_connections(menv, params['agents'])

    sim = CollabSimulation(menv,
                           precallback=menv.match_collab_partners,
                           callback=menv.post_cbk,
                           log_folder=log_folder)

    # RUN SIMULATION
    for i in range(num_of_steps):
        sim.async_step()

    menv.save_artifact_info(path)
    sim.end()

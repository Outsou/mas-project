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
import time


if __name__ == "__main__":
    # DEFINE TEST PARAMETERS
    num_of_steps = 400
    pop_size = 20
    shape = (64, 64)
    sample_size = 8
    #create_kwargs = coe.get_create_kwargs(20, (64, 64))
    menv = coe.create_environment(num_of_slaves=8)
    params = coe.DEFAULT_PARAMS
    params['agents'] = 10
    params['novelty_weight'] = 0.2
    # params['model'] = 'Q'

    # END PARAM DEF

    # CREATE SIMULATION AND RUN
    path = 't033_a{}_e{}_i{}'\
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
    step_times = []
    for i in range(num_of_steps):
        step_start = time.monotonic()
        sim.async_step()
        step_time = time.monotonic() - step_start
        step_times.append(step_time)
        mean_step_time = np.mean(step_times)
        run_end_time = time.ctime(time.time() + (mean_step_time * (num_of_steps - (i + 1))))
        print('Step {}/{} finished in {:.3f} seconds. Estimated run end time at: {}'
              .format((i + 1), num_of_steps, step_time, run_end_time))

    menv.save_artifact_info(path)
    sim.end()

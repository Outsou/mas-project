"""Testing genetic image generation with one agent. Images from agent's memory
are saved at the end of the run.
"""
import numpy as np
from matplotlib import cm

from artifacts import GeneticImageArtifact
from utils.util import create_toolbox, create_pset, create_environment, get_image_rules

from creamas.core import Simulation

import aiomas
import os
import shutil


def _make_rules(rule_names, shape):
    rule_dict = get_image_rules(shape)
    rules = []
    for name in rule_names:
        rules.append(rule_dict[name])
    return rules


def _init_save_folder(save_folder):
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)


def run_sim(menv, steps, log_folder, save_folder, pset, output_shape):
    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(steps)
    """
    if save_folder is not None:
        cm_name = 'viridis'
        x = np.linspace(0.0, 1.0, 256)
        color_map = cm.get_cmap(cm_name)(x)[np.newaxis, :, :3][0]
        menv.save_artifacts(save_folder, pset, color_map, output_shape)
    """
    sim.end()


def test_fitness_rules(rule_names,
                       weights,
                       steps=10,
                       pop_size=20,
                       search_width=10,
                       log_folder=None,
                       save_folder=None,
                       internal_shape=(64, 64),
                       output_shape=(400, 400)):
    """Test GP image generation with given aesthetic (fitness) rules.
    """
    critic_threshold = 0.000
    veto_threshold = 0.000
    novelty_weight = -1
    memsize = 10000
    shape = internal_shape

    pset = create_pset()
    create_kwargs = {'pset': pset,
                     'toolbox': create_toolbox(pset),
                     'pop_size': pop_size,
                     'shape': shape}

    # Make the rules
    rules = _make_rules(rule_names, shape)
    rule_weights = weights
    _init_save_folder(save_folder)

    # Environment and simulation
    menv = create_environment(2)

    for _ in range(1):
        ret = aiomas.run(until=menv.spawn('agents:GeneticImageAgent',
                                          log_folder=log_folder,
                                          save_folder=save_folder,
                                          artifact_cls=GeneticImageArtifact,
                                          create_kwargs=create_kwargs,
                                          rules=rules,
                                          rule_weights=rule_weights,
                                          memsize=memsize,
                                          critic_threshold=critic_threshold,
                                          veto_threshold=veto_threshold,
                                          novelty_weight=novelty_weight,
                                          search_width=search_width,
                                          output_shape=output_shape))

    run_sim(menv, steps, log_folder, save_folder, pset, output_shape)


if __name__ == "__main__":
    test_fitness_rules(('symm', 'benford', 'fd_aesthetics'),
                       (1.0, 1.0, 1.0),
                       steps=40,
                       pop_size=20,
                       search_width=10,
                       log_folder='gp_test_logs',
                       save_folder='gp_test_symm_ben_comp')


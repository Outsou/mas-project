'''Testing genetic image generation with one agent. Images from agent's memory are saved at the end of the run.'''

from artifacts import GeneticImageArtifact
from utils.util import create_toolbox, create_pset, create_environment, get_image_rules

from creamas.core import Simulation

import aiomas
import os
import shutil


if __name__ == "__main__":

    # Parameters

    critic_threshold = 0.001
    veto_threshold = 0.001

    novelty_weight = 0.85
    memsize = 100000

    pset = create_pset()

    shape = (32, 32)
    create_kwargs = {'pset': pset,
                     'toolbox': create_toolbox(pset),
                     'pop_size': 10,
                     'shape': shape}

    # Make the rules

    rule_dict = get_image_rules(shape)

    rules = []
    rule_weights = []
    rules.append(rule_dict['red'])
    rule_weights.append(1)


    # Environment and simulation

    log_folder = 'gp_test_logs'
    save_folder = 'gp_test_artifacts'
    if os.path.exists(save_folder):
        shutil.rmtree(save_folder)
    os.makedirs(save_folder)

    menv = create_environment(2)

    for _ in range(1):
        ret = aiomas.run(until=menv.spawn('agents:FeatureAgent',
                                          log_folder=log_folder,
                                          artifact_cls=GeneticImageArtifact,
                                          create_kwargs=create_kwargs,
                                          rules=rules,
                                          rule_weights=rule_weights,
                                          memsize=memsize,
                                          critic_threshold=critic_threshold,
                                          veto_threshold=veto_threshold,
                                          novelty_weight=novelty_weight))

        print(ret)

    sim = Simulation(menv, log_folder=log_folder)
    sim.async_steps(10)
    menv.save_artifacts(save_folder)
    sim.end()


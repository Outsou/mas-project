"""Experiment for how different regularization weights affect learning."""
from experiments.multi_learner_base import run_experiment

if __name__ == "__main__":
    params = {'change_speed': 0} # Tested with 0 and 0.05
    loop = ('reg_weight', [0.1, 0.5, 0.9])

    num_of_simulations = 50
    num_of_steps = 1000

    run_experiment(params, loop, num_of_simulations, num_of_steps)

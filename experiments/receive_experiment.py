"""Experiment with the active agent also receiving artifacts."""
from experiments.multi_learner_base import run_experiment

if __name__ == "__main__":
    params = {'send_prob': [0, 0.2, 0.6, 1]}

    num_of_simulations = 50
    num_of_steps = 1000

    run_experiment(params, num_of_simulations, num_of_steps)
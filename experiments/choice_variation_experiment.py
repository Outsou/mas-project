"""Experiment for how much the models pick the same agent."""
from experiments.multi_learner_base import run_experiment

if __name__ == "__main__":
    params = {'search_width': [1, 10, 100, 1000]}

    num_of_simulations = 50
    num_of_steps = 1000

    run_experiment(params, num_of_simulations, num_of_steps)



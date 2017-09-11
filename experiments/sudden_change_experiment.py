"""Experiment testing how a sudden change in preferences affects the learning models."""
from experiments.multi_learner_base import run_experiment

if __name__ == "__main__":
    params = {'instant_steps': 500} # Tested with 0 and 0.05
    loop = ('instant_amount', [0.1, 0.3, 0.5])

    num_of_simulations = 10
    num_of_steps = 500

    run_experiment(params, loop, num_of_simulations, num_of_steps, draw_windows=True)

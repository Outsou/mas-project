"""Experiment with preferences changing at every time step."""
from experiments.multi_learner_image_base import run_experiment

if __name__ == "__main__":
    params = {'search_width': 10}

    num_of_simulations = 20
    num_of_steps = 200

    run_experiment(params, num_of_simulations, num_of_steps)
"""Experiment with preferences changing at every time step."""
from experiments.multi_learner_base import run_experiment

if __name__ == "__main__":
    params = {'change_speed': [0, 0.01, 0.05, 0.1]}

    num_of_simulations = 50
    num_of_steps = 1000

    run_experiment(params, num_of_simulations, num_of_steps)
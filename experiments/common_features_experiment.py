"""Experiment with the active agent being able to observe 1-5 of the features all other agents observe"""
from experiments.multi_learner_base import run_experiment

if __name__ == "__main__":
    params = {'common_features': list(range(1, 6))} # range from 1 to 5

    num_of_simulations = 50
    num_of_steps = 1000

    run_experiment(params, num_of_simulations, num_of_steps)

"""Test with all agents modeling each other with different amount of common features."""
from experiments.multi_learner_base2 import run_experiment

if __name__ == "__main__":
    params = {'common_features': [1, 3, 5]}

    num_of_simulations = 25
    num_of_steps = 1000

    run_experiment(params, num_of_simulations, num_of_steps)

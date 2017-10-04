""""""
from experiments.single_learner_base import run_experiment

if __name__ == "__main__":
    params = {'model': ['Q', 'linear', 'sgd']}

    num_of_simulations = 50
    num_of_steps = 1000

    run_experiment(params, num_of_simulations, num_of_steps)

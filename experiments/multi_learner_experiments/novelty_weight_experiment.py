"""
Experiment to see how the different models adapt to novelty being a factor in evaluation.
Novelty isn't given separately in the evaluation.
"""
from experiments.multi_learner_base import run_experiment

if __name__ == "__main__":
    params = {'memsize': 32,
              'novelty_weight': [0, 0.2, 0.5, 0.8]}

    num_of_simulations = 50
    num_of_steps = 1000

    run_experiment(params, num_of_simulations, num_of_steps)



'''This module contains different types of agents.'''

from creamas.rules.agent import RuleAgent
from creamas.math import gaus_pdf

import logging
import aiomas
import numpy as np


class FeatureAgent(RuleAgent):
    '''The base class for agents that use features.
    Should work with any artifact class that implements distance, max_distance, and invent functions.'''
    def __init__(self, environment, artifact_cls, create_kwargs, rules, rule_weights = None,
                 novelty_weight=0.5, search_width=10, critic_threshold=0.5, veto_threshold=0.5,
                 log_folder=None, log_level=logging.INFO, memsize=0):
        '''

        :param environment:
            Agent's environment.
        :param artifact_cls:
            Artifact class for the agent to use.
        :param create_kwargs:
            Parameters needed to create an artifact from class artifact_cls.
        :param rules:
            Rules used to evaluate an artifact.
        :param rule_weights:
            Weights for the rules used to evaluate an artifact.
        :param novelty_weight:
            How much novelty weights in artifact evaluation w.r.t value.
        :param search_width:
            How many artifacts are created when inventing.
        :param critic_threshold:
            Threshold for own artifacts.
        :param veto_threshold:
            Threshold for other agents' artifacts.
        :param log_folder:
        :param log_level:
        :param memsize:
            Size of agent's memory
        '''
        super().__init__(environment, log_folder=log_folder,
                         log_level=log_level)

        max_distance = artifact_cls.max_distance(create_kwargs)
        self.stmem = FeatureAgent.STMemory(artifact_cls=artifact_cls, length=memsize, max_length=memsize, max_distance=max_distance)
        self.artifact_cls = artifact_cls
        self._own_threshold = critic_threshold
        self._veto_threshold = veto_threshold
        self.search_width = search_width
        self.create_kwargs = create_kwargs
        self.novelty_weight = novelty_weight

        if rule_weights is None:
            rule_weights = [1] * len(rules)
        else:
            assert len(rules) == len(rule_weights), "Different amount of rules and rule weights."

        for i in range(len(rules)):
            self.add_rule(rules[i], rule_weights[i])

    def novelty(self, artifact):
        '''Novelty of an artifact w.r.t agent's memory.'''
        distance = self.stmem.distance(artifact)
        return distance

    @aiomas.expose
    def evaluate(self, artifact):
        '''Evaluates an artifact based on value and novelty'''
        if self.name in artifact.evals:
            return artifact.evals[self.name], None

        value, _ = super().evaluate(artifact)
        if self.stmem.length <= 0:
            artifact.add_eval(self, value)
            return value, None

        novelty = self.novelty(artifact)
        evaluation = (1 - self.novelty_weight) * value + self.novelty_weight * novelty
        artifact.add_eval(self, evaluation)

        return evaluation, None

    def invent(self, n):
        '''Invents an artifact with n iterations and chooses the best.'''
        best_artifact, _ = self.artifact_cls.invent(n, self, self.create_kwargs)
        return best_artifact

    def learn(self, artifact, iterations=1):
        '''Adds an artifact to memory.'''
        for i in range(iterations):
            self.stmem.train_cycle(artifact)

    @aiomas.expose
    def get_addr(self):
        return self.addr

    @aiomas.expose
    async def get_criticism(self, artifact):
        '''Returns True if artifact passes agent's threshold, or False otherwise.'''
        evaluation, _ = self.evaluate(artifact)

        if evaluation >= self._veto_threshold:
            #self.learn(artifact, self.teaching_iterations)
            return True, artifact
        else:
            return False, artifacts

    @aiomas.expose
    def get_name(self):
        return self.name

    @aiomas.expose
    async def act(self):
        # Create and evaluate an artifact
        artifact = self.invent(self.search_width)
        eval, _ = self.evaluate(artifact)
        self._log(logging.INFO, 'Created artifact with evaluation {}'.format(eval))
        self.add_artifact(artifact)

        if eval >= self._own_threshold:
            self.learn(artifact)

    @aiomas.expose
    def save_artifacts(self, folder):
        i = 0
        for art in reversed(self.stmem.artifacts[:self.stmem.length]):
            i += 1
            self.artifact_cls.save_artifact(art, folder, i, art.evals[self.name])


    class STMemory:

        '''Agent's short-term memory model using a simple list which stores
        artifacts as is.'''
        def __init__(self, artifact_cls, length, max_distance, max_length = 100):
            self.length = length
            self.artifacts = []
            self.max_length = max_length
            self.artifact_cls = artifact_cls
            self.max_distance = max_distance

        def _add_artifact(self, artifact):
            if len(self.artifacts) >= 2 * self.max_length:
                self.artifacts = self.artifacts[:self.max_length]
            self.artifacts.insert(0, artifact)

        def learn(self, artifact):
            '''Learn new artifact. Removes last artifact from the memory if it is
            full.'''
            self._add_artifact(artifact)

        def train_cycle(self, artifact):
            '''Train cycle method to keep the interfaces the same with the SOM
            implementation of the short term memory.
            '''
            self.learn(artifact)

        def distance(self, artifact):
            limit = self.get_comparison_amount()
            if limit == 0:
                return np.random.random() * self.max_distance / self.max_distance
            min_distance = self.max_distance
            for a in self.artifacts[:limit]:
                d = self.artifact_cls.distance(artifact, a)
                if d < min_distance:
                    min_distance = d
            return min_distance / self.max_distance

        def get_comparison_amount(self):
            if len(self.artifacts) < self.length:
                amount = len(self.artifacts)
            else:
                amount = self.length
            return amount


class MultiAgent(FeatureAgent):
    '''An agent that learns multiple models at the same time.
    Used for testing and comparing different modeling methods.
    An artifact is evaluated with a weighed sum of its features' distances from the preferred values.
    The distance is calculated using a gaussian distribution's pdf.'''
    def __init__(self, environment, std, active=False, *args, **kwargs):
        '''
        :param std:
            Standard deviation for the gaussian distribution used in evaluation.
        :param active:
            Agent acts only if it is active.
        '''
        super().__init__(environment, *args, **kwargs)
        self.std = std
        self.sgd_reward = 0     # reward with stochastic gradient descent
        self.bandit_reward = 0  # reward with q-learning
        self.random_reward = 0  # expected random reward
        self.max_reward = 0     # maximum possible reward
        self.active = active
        self.connection_totals = {} # contains max reward from a single connection

    def get_features(self, artifact):
        '''Return objective values for features without mapping.'''
        features = []
        for rule in self.R:
            features.append(rule.feat(artifact))
        return features

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)

        # Initialize the multi-model learner
        self.learner = MultiAgent.MultiLearner(list(self.connections), len(self.R), self.std)

        for connection in self.connections.keys():
            self.connection_totals[connection] = 0

        return rets

    @aiomas.expose
    async def act(self):
        '''If active, create an artifact and send it to everyone for evaluation.
        Update models based on the evaluation received from the connection chosen by the model.'''
        if not self.active:
            return

        # Create an artifact
        artifact, _ = self.artifact_cls.invent(self.search_width, self, self.create_kwargs)
        features = self.get_features(artifact)
        eval, _ = self.evaluate(artifact)

        # Gather evaluations from the other agents
        opinions = {}
        for addr in list(self.connections.keys()):
            remote_agent = await self.env.connect(addr)
            opinion, _ = await remote_agent.evaluate(artifact)
            opinions[addr] = opinion
            self.connection_totals[addr] += opinion

        # Stochastic gradient descent selection and update
        sgd_chosen_addr = self.learner.sgd_choose(features)
        self.sgd_reward += opinions[sgd_chosen_addr]
        self.learner.update_sgd(opinions[sgd_chosen_addr], sgd_chosen_addr, features)

        # Q-learning selection and update
        bandit_chosen_addr = self.learner.bandit_choose()
        self.bandit_reward += opinions[bandit_chosen_addr]
        self.learner.update_bandit(opinions[bandit_chosen_addr], bandit_chosen_addr)

        # Update expected random reward and max reward
        self.random_reward += np.sum(list(opinions.values())) / len(opinions)
        self.max_reward += opinions[max(opinions, key=opinions.get)]

    @aiomas.expose
    def close(self, folder=None):
        if not self.active:
            return

        # Log stats about the run
        self._log(logging.INFO, 'SGD reward: ' + str(self.sgd_reward))
        self._log(logging.INFO, 'Bandit reward: ' + str(self.bandit_reward))
        self._log(logging.INFO, 'Max reward: ' + str(self.max_reward))
        self._log(logging.INFO, 'Expected random reward: ' + str(self.random_reward))
        self._log(logging.INFO, 'Connection totals:')
        for addr, total in self.connection_totals.items():
            self._log(logging.INFO, '{}: {}'.format(addr, total))
        self._log(logging.INFO, 'Bandit perceived as best: ' + str(self.learner.bandit_choose(e=0)))


    class MultiLearner():
        '''A learner that is capable of using and updating multiple models.
        Currently implemented:
        stochastic gradient descent (SGD),
        Q-learning for n-armed bandit problem'''
        def __init__(self, addrs, num_of_features, std, centroid_rate=200, weight_rate=0.2):
            '''
            :param addrs:
                Addresses of the agents that are modeled.
            :param num_of_features:
                Number of features in an artifact.
            :param std:
                Standard deviation for evaluating artifacts.
            :param centroid_rate:
                Learning rate for centroids.
            :param weight_rate:
                Learning rate for weights.
            '''
            self.centroid_rate = centroid_rate
            self.weight_rate = weight_rate
            self.num_of_features = num_of_features
            self.std = std

            self.sgd_weights = {}
            self.centroids = {}
            self.bandits = {}
            for addr in addrs:
                self.sgd_weights[addr] = np.array([0.5] * num_of_features)
                self.centroids[addr] = np.array([0.5] * num_of_features)
                self.bandits[addr] = 10

            self.max = gaus_pdf(1, 1, std)

        def _sgd_estimate(self, addr, features):
            '''Estimate value the SGD model.
            Returns the estimate and individual values for different features.'''
            vals = np.zeros(self.num_of_features)
            for i in range(self.num_of_features):
                vals[i] = gaus_pdf(features[i], self.centroids[addr][i], self.std) / self.max
            estimate = np.sum(self.sgd_weights[addr] * vals)
            return estimate, vals

        def update_bandit(self, true_value, addr, discount_factor = 0.95, learning_factor=0.9):
            '''
            Updates the Q-value for addr.

            :param true_value:
                The evaluation of the artifact from addr.
            :param addr:
                The connection that will be updated.
            :param discount_factor:
                Controls how much the reward from future is discounted.
            :param learning_factor:
                Controls learning speed.
            '''
            old_value = self.bandits[addr]
            self.bandits[addr] += learning_factor * (true_value + discount_factor * old_value - old_value)

        def update_sgd(self, true_value, addr, features):
            '''
            Uses SGD to update weights and centroid for the SGD model.
            :param true_value:
                The evaluation of the artifact from addr.
            :param addr:
                The connection that will be updated.
            :param features:
                Objective features of an artifact.
            '''
            estimate, vals = self._sgd_estimate(addr, features)

            error = true_value - estimate

            # Update weights
            gradient = vals * error
            self.sgd_weights[addr] += self.weight_rate * gradient
            self.sgd_weights[addr] = np.clip(self.sgd_weights[addr], 0, 1)

            # Calculate gradient of gaussian pdf w.r.t mean
            gradient = (features - self.centroids[addr]) \
                       * np.exp(-(features - self.centroids[addr]) ** 2 / (2 * self.std ** 2)) \
                       / np.sqrt(2 * np.pi) * (self.std ** 2) ** (3 / 2)

            # Update centroid
            self.centroids[addr] += self.centroid_rate * gradient * error
            self.centroids[addr] = np.clip(self.centroids[addr], 0, 1)

        def sgd_choose(self, features, e=0.2):
            '''
            Choose a connection with the SGD model.

            :param features:
                Objective features of an artifact.
            :param e:
                Chance that a random connection will be chosen.
            :return:
                The chosen connection's address.
            '''
            if np.random.rand() < e:
                return np.random.choice(list(self.sgd_weights.keys()))

            best_estimate = -1
            best_addr = None

            for addr in self.sgd_weights.keys():
                estimate, _ = self._sgd_estimate(addr, features)
                if estimate > best_estimate:
                    best_estimate = estimate
                    best_addr = addr

            return best_addr

        def bandit_choose(self, e=0.2):
            '''
            Choose a connection with the Q-learning model.

            :param features:
                Objective features of an artifact.
            :param e:
                Chance that a random connection will be chosen.
            :return:
                The chosen connection's address.
            '''
            if np.random.rand() < e:
                return np.random.choice(list(self.bandits.keys()))

            best = -np.inf
            best_addr = None

            for addr, value in self.bandits.items():
                if value > best:
                    best = value
                    best_addr = addr

            return best_addr


from creamas.rules.agent import RuleAgent
from creamas.math import gaus_pdf

import logging
import aiomas
import numpy as np
import matplotlib.pyplot as plt


class FeatureAgent(RuleAgent):
    def __init__(self, environment, artifact_cls, create_kwargs, rules, rule_weights = None, desired_novelty=-1,
                 novelty_weight=0.5, ask_criticism=True, search_width=10, ask_random=False,
                 critic_threshold=0.5, veto_threshold=0.5, log_folder=None,
                 log_level=logging.INFO, memsize=0, hedonic_std=0.1):
        super().__init__(environment, log_folder=log_folder,
                         log_level=log_level)

        max_distance = artifact_cls.max_distance(create_kwargs)
        self.stmem = FeatureAgent.STMemory(artifact_cls=artifact_cls, length=memsize, max_length=memsize, max_distance=max_distance)
        self.artifact_cls = artifact_cls
        self._own_threshold = critic_threshold
        self._novelty_threshold = veto_threshold
        self.search_width = search_width
        self.ask_random = ask_random
        self.ask_criticism = ask_criticism
        self.desired_novelty = desired_novelty
        self.hedonic_std = hedonic_std
        self.create_kwargs = create_kwargs
        self.novelty_weight = novelty_weight

        self.connection_counts = None
        self.connection_list = []

        self.comparison_count = 0
        self.artifacts_created = 0
        self.passed_self_criticism_count = 0

        if rule_weights is None:
            rule_weights = [1] * len(rules)
        else:
            assert len(rules) == len(rule_weights), "Different amount of rules and rule weights."

        for i in range(len(rules)):
            self.add_rule(rules[i], rule_weights[i])

    def novelty(self, artifact):
        self.comparison_count += self.stmem.get_comparison_amount()
        distance = self.stmem.distance(artifact)
        return distance

    def hedonic_value(self, value, desired_value):
        lmax = gaus_pdf(desired_value, desired_value, self.hedonic_std)
        pdf = gaus_pdf(value, desired_value, self.hedonic_std)
        return pdf / lmax

    @aiomas.expose
    def evaluate(self, artifact):
        if self.name in artifact.evals:
            return artifact.evals[self.name], None

        value, _ = super().evaluate(artifact)
        if self.stmem.length <= 0:
            artifact.add_eval(self, value)
            return value, None

        novelty = self.novelty(artifact)
        evaluation = (1 - self.novelty_weight) * value + self.novelty_weight * novelty
        artifact.add_eval(self, evaluation)
        eval_values = {'value': value, 'novelty': novelty}
        artifact.framings['eval_values'] = eval_values

        return evaluation, eval_values

    def invent(self, n):
        best_artifact, _ = self.artifact_cls.invent(n, self, self.create_kwargs)
        return best_artifact

    def learn(self, artifact, iterations=1):
        for i in range(iterations):
            self.stmem.train_cycle(artifact)

    @aiomas.expose
    def get_addr(self):
        return self.addr

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)

        self.connection_counts = {}
        for conn in conns:
            self.connection_counts[conn[0]] = 0

        return rets

    @aiomas.expose
    async def get_criticism(self, artifact):
        evaluation, _ = self.evaluate(artifact)

        if evaluation >= self._novelty_threshold:
            #self.learn(artifact, self.teaching_iterations)
            return True, artifact
        else:
            return False, artifact

    @aiomas.expose
    def get_connection_counts(self):
        return self.connection_counts

    @aiomas.expose
    def get_comparison_count(self):
        return self.comparison_count

    @aiomas.expose
    def get_artifacts_created(self):
        return self.artifacts_created

    @aiomas.expose
    def get_name(self):
        return self.name

    @aiomas.expose
    def get_desired_novelty(self):
        return self.desired_novelty

    @aiomas.expose
    def get_passed_self_criticism_count(self):
        return self.passed_self_criticism_count

    @aiomas.expose
    async def act(self):
        artifact = self.invent(self.search_width)
        eval, eval_values = self.evaluate(artifact)
        # artifact.framings['eval_values'] = eval_values
        self._log(logging.INFO, 'Created artifact with evaluation {}'.format(eval))
        self.add_artifact(artifact)

        if eval >= self._own_threshold:
            artifact.self_criticism = 'pass'
            self.passed_self_criticism_count += 1
            if self.stmem.length > 0 and eval_values['novelty'] > 0:
                self.learn(artifact)

    @aiomas.expose
    def save_artifacts(self, folder):
        i = 0
        for art in reversed(self.stmem.artifacts[:self.stmem.length]):
            i += 1
            plt.imshow(art.obj, shape=art.obj.shape)
            plt.title('Eval: {}'.format(art.evals[self.name]))
            plt.savefig('{}/artifact{}'.format(folder, i))


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

    def __init__(self, environment, std, active=False, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.std = std
        self.sgd_reward = 0
        self.bandit_reward = 0
        self.random_reward = 0
        self.max_reward = 0
        self.active = active
        self.connection_totals = {}

    def get_features(self, artifact):
        features = []
        for rule in self.R:
            features.append(rule.feat(artifact))
        return features

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)
        self.learner = MultiAgent.MultiLearner(list(self.connections), len(self.R), self.std)

        for connection in self.connections.keys():
            self.connection_totals[connection] = 0

        return rets

    @aiomas.expose
    def give_artifact(self, artifact, eval, addr):
        features = self.get_features(artifact)
        self.learner.give_sample(eval, addr, features)

    @aiomas.expose
    async def act(self):
        if not self.active:
            return

        artifact, _ = self.artifact_cls.invent(self.search_width, self, self.create_kwargs)
        features = self.get_features(artifact)
        eval, _ = self.evaluate(artifact)
        # print(eval)

        if not self.active:
            # addr = list(self.connections.keys())[0]
            # remote_agent = await self.env.connect(addr)
            # await remote_agent.give_artifact(artifact, eval, self.addr)
            pass
        else:
            opinions = {}
            for addr in list(self.connections.keys()):
                remote_agent = await self.env.connect(addr)
                opinion, _ = await remote_agent.evaluate(artifact)
                opinions[addr] = opinion
                self.connection_totals[addr] += opinion

            # print(opinions)
            # import time
            # time.sleep(1)

            sgd_chosen_addr = self.learner.sgd_choose(features)
            self.sgd_reward += opinions[sgd_chosen_addr]
            self.learner.update_sgd(opinions[sgd_chosen_addr], sgd_chosen_addr, features)

            bandit_chosen_addr = self.learner.bandit_choose()
            self.bandit_reward += opinions[bandit_chosen_addr]
            self.learner.update_bandit(opinions[bandit_chosen_addr], bandit_chosen_addr)

            self.random_reward += np.sum(list(opinions.values())) / len(opinions)
            self.max_reward += opinions[max(opinions, key=opinions.get)]

    @aiomas.expose
    def close(self, folder=None):
        if not self.active:
            return

        self._log(logging.INFO, 'SGD reward: ' + str(self.sgd_reward))
        self._log(logging.INFO, 'Bandit reward: ' + str(self.bandit_reward))
        self._log(logging.INFO, 'Max reward: ' + str(self.max_reward))
        self._log(logging.INFO, 'Random reward: ' + str(self.random_reward))
        self._log(logging.INFO, 'Connection totals:')
        for addr, total in self.connection_totals.items():
            self._log(logging.INFO, '{}: {}'.format(addr, total))
        self._log(logging.INFO, 'Bandit perceived as best: ' + str(self.learner.bandit_choose(e=0)))


    class MultiLearner():
        def __init__(self, addrs, num_of_features, std, centroid_rate=200, weight_rate=0.2):
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
            vals = np.zeros(self.num_of_features)
            for i in range(self.num_of_features):
                vals[i] = gaus_pdf(features[i], self.centroids[addr][i], self.std) / self.max
            estimate = np.sum(self.sgd_weights[addr] * vals)
            return estimate, vals

        def update_bandit(self, true_value, addr, discount_factor = 0.95, learning_factor=0.9):
            old_value = self.bandits[addr]
            self.bandits[addr] += learning_factor * (true_value + discount_factor * old_value - old_value)

        def update_sgd(self, true_value, addr, features):
            estimate, vals = self._sgd_estimate(addr, features)

            error = true_value - estimate

            # Update weights
            gradient = vals * error
            self.sgd_weights[addr] += self.weight_rate * gradient
            self.sgd_weights[addr] = np.clip(self.sgd_weights[addr], 0, 1)

            # Calculate gradient of gaus pdf w.r.t mean
            gradient = (features - self.centroids[addr]) \
                       * np.exp(-(features - self.centroids[addr]) ** 2 / (2 * self.std ** 2)) \
                       / np.sqrt(2 * np.pi) * (self.std ** 2) ** (3 / 2)

            # Update centroid
            self.centroids[addr] += self.centroid_rate * gradient * error
            self.centroids[addr] = np.clip(self.centroids[addr], 0, 1)

        def sgd_choose(self, features, e=0.2):
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
            if np.random.rand() < e:
                return np.random.choice(list(self.bandits.keys()))

            best = -np.inf
            best_addr = None

            for addr, value in self.bandits.items():
                if value > best:
                    best = value
                    best_addr = addr

            return best_addr


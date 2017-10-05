'''This module contains different types of agents.'''

from creamas.rules.agent import RuleAgent
from creamas.rules.rule import RuleLeaf
from creamas.mappers import GaussianMapper

from artifacts import GeneticImageArtifact as GIA
from learners import MultiLearner

import logging
import aiomas
import numpy as np
import copy
import pickle
import os


def record_stats(stats, key, addr, reward, chose_best):
    """
    Used to record stats at each step for MultiAgent

    :param stats:
        Stats-dictionary.
    :param key:
        Name of the learning method.
    :param addr:
        Address of the connection chosen by the learning method.
    :param reward:
        Reward from addr.
    :param chose_best:
        True if optimal choice was made, else False.
    """
    stats[key]['connections'].append(addr)
    stats[key]['rewards'].append(reward)
    if chose_best:
        stats[key]['chose_best'].append(1)
    else:
        stats[key]['chose_best'].append(0)


def agent_name_parse(name):
    '''Converts the name of an agent into a file path friendly format.'''
    parsed_name = name.replace('://', '_')
    parsed_name = parsed_name.replace(':', '_')
    parsed_name = parsed_name.replace('/', '_')
    return parsed_name


class FeatureAgent(RuleAgent):
    '''The base class for agents that use features.
    Should work with any artifact class that implements distance, max_distance,
    and invent functions.'''
    def __init__(self, environment, artifact_cls, create_kwargs, rules,
                 rule_weights=None, novelty_weight=0.5, search_width=10,
                 critic_threshold=0.5, veto_threshold=0.5,
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
        observed_features = [str(rule.feat) for rule in rules]
        self.stmem = FeatureAgent.STMemory(artifact_cls=artifact_cls,
                                           length=memsize,
                                           max_length=memsize,
                                           features=observed_features)
        self.artifact_cls = artifact_cls
        self._own_threshold = critic_threshold
        self._veto_threshold = veto_threshold
        self.search_width = search_width
        self.create_kwargs = create_kwargs
        self.novelty_weight = novelty_weight

        if rule_weights is None:
            rule_weights = [1] * len(rules)
        else:
            assert len(rules) == len(rule_weights),\
                "Different amount of rules and rule weights."

        for i in range(len(rules)):
            self.add_rule(rules[i], rule_weights[i])


    def novelty(self, artifact):
        '''Novelty of an artifact w.r.t agent's memory.'''
        distance = self.stmem.distance(artifact)
        return distance

    @aiomas.expose
    def evaluate(self, artifact):
        """Evaluates an artifact based on value and novelty.
        """
        if self.name in artifact.evals:
            return artifact.evals[self.name], artifact.framings[self.name]

        value, _ = super().evaluate(artifact)
        if self.novelty_weight == -1:
            fr = {'value': value, 'novelty': None}
            artifact.add_eval(self, value, fr)
            return value, fr

        novelty = self.novelty(artifact)
        evaluation = (1 - self.novelty_weight) * value + self.novelty_weight * novelty
        fr = {'value': value, 'novelty': novelty}
        artifact.add_eval(self, evaluation, fr)

        return evaluation, fr

    def invent(self, n, *args, **kwargs):
        """Invents an artifact with n iterations and chooses the best.
        """
        best_artifacts = self.artifact_cls.invent(n, self, self.create_kwargs,
                                                  *args, **kwargs)
        return best_artifacts

    def learn(self, artifact, iterations=1):
        """Adds an artifact to memory.
        """
        for i in range(iterations):
            self.stmem.train_cycle(artifact)

    @aiomas.expose
    def get_addr(self):
        return self.addr

    @aiomas.expose
    async def get_criticism(self, artifact):
        """Returns True if artifact passes agent's threshold, or False
        otherwise.
        """
        evaluation, _ = self.evaluate(artifact)

        if evaluation >= self._veto_threshold:
            self.learn(artifact)
            return True, artifact
        else:
            return False, artifact

    @aiomas.expose
    def get_name(self):
        return self.name

    @aiomas.expose
    async def act(self):
        # Create and evaluate an artifact
        artifacts = self.invent(self.search_width, n_artifacts=1)
        for artifact, _ in artifacts:
            e, fr = self.evaluate(artifact)
            n = None if fr['novelty'] is None else np.around(fr['novelty'], 2)
            self._log(logging.INFO,
                      'Created artifact with evaluation {} (v: {}, n: {})'
                      .format(np.around(e, 2), np.around(fr['value'], 2), n))
            self.add_artifact(artifact)

            if e >= self._own_threshold:
                self.learn(artifact)

    @aiomas.expose
    def get_artifacts(self):
        artifacts = list(reversed(self.stmem.artifacts[:self.stmem.length]))
        return artifacts, self.artifact_cls

    @aiomas.expose
    def get_log_folder(self):
        return self.logger.folder


    class STMemory:
        """Agent's short-term memory model using a simple list which stores
        artifacts as is.
        """
        def __init__(self, artifact_cls, length, features, max_length=100):
            self.length = length
            self.artifacts = []
            self.max_length = max_length
            self.artifact_cls = artifact_cls
            # Max distance is between a vector of zeros and a vector of ones
            self.max_distance = np.linalg.norm(np.ones(len(features)))
            self.features = features

        def _add_artifact(self, artifact):
            if len(self.artifacts) >= 2 * self.max_length:
                self.artifacts = self.artifacts[:self.max_length]
            self.artifacts.insert(0, artifact)

        def learn(self, artifact):
            """Learn new artifact. Removes last artifact from the memory if it
            is full.
            """
            self._add_artifact(artifact)

        def train_cycle(self, artifact):
            """Train cycle method to keep the interfaces the same with the SOM
            implementation of the short term memory.
            """
            self.learn(artifact)

        def distance(self, artifact):
            def extract_observed_features(art):
                observed_features = []
                for feature in self.features:
                    observed_features.append(art.framings['features'][feature])
                return np.array(observed_features)

            limit = self.get_comparison_amount()
            if limit == 0:
                return np.random.random() * self.max_distance / self.max_distance
            min_distance = self.max_distance

            features = extract_observed_features(artifact)
            for a in self.artifacts[:limit]:
                d = np.linalg.norm(features - extract_observed_features(a))
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
    """An agent that learns multiple models at the same time.
    Used for testing and comparing different modeling methods.
    An artifact is evaluated with a weighed sum of its features' distances from
    the preferred values. The distance is calculated using a gaussian
    distribution's pdf.
    """
    def __init__(self, environment, data_folder, std=None, active=False,
                 rule_vec=None, reg_weight=0.1, gaussian_updates=False,
                 send_prob=0, own_folder=False, *args, **kwargs):
        """
        :param std:
            Standard deviation for the gaussian distribution used in evaluation.
        :param bool active:
            Agent acts only if it is active.
        :param rule_vec:
            Controls the change in agent's preferences. Is added to centroids
            each step.
        :param bool gaussian_updates:
            True if agent preferences are updated drawing from Gaussian
            distributions.
        :param float send_prob:
            Probability, that an inactive agent
        """
        super().__init__(environment, *args, **kwargs)
        self.std = std
        self.active = active
        self.age = 0
        stat_dict = {'connections': [], 'rewards': [], 'chose_best': []}
        self.stats = {'bandit': copy.deepcopy(stat_dict),
                      'linear': copy.deepcopy(stat_dict),
                      'poly': copy.deepcopy(stat_dict),
                      'random_rewards': [],
                      'max_rewards': [],
                      'opinions': []}
        if std is not None:
            self.stats['sgd'] =  copy.deepcopy(stat_dict)
        self.learner = None
        self.data_folder = data_folder
        self.reg_weight = reg_weight
        self.gaussian_updates = gaussian_updates
        self.send_prob = send_prob
        self.own_folder = own_folder

        if rule_vec is not None:
            assert len(rule_vec) == len(self.R), \
                'Length of rule_vec differs from the amount of rules.'

        self.rule_vec = rule_vec
        self.gaussian_updates = gaussian_updates

    def get_features(self, artifact):
        """Return objective values for features without mapping.
        """
        features = []
        for rule in self.R:
            features.append(rule.feat(artifact))
        return features

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)

        # Initialize the multi-model learner
        self.learner = MultiLearner(list(self.connections),
                                    len(self.R),
                                    self.std,
                                    reg_weight=self.reg_weight)

        return rets

    @aiomas.expose
    def cause_change(self, amount):
        if self.active:
            return

        for i in range(len(self.R)):
            mean = self.R[i].mapper._mean
            if mean <= 0.5:
                mean += amount
            else:
                mean -= amount
            mean = np.clip(mean, 0, 1)
            self._R[i] = RuleLeaf(self.R[i].feat, GaussianMapper(mean, self.std))

    def update_means(self):
        """Update means for rules
        """
        if self.std is None:
            return

        for i in range(len(self.R)):
            if self.gaussian_updates:
                mean = self.R[i].mapper._mean +\
                       np.random.normal(0.0, self.rule_vec[i])
                mean = np.clip(mean, 0, 1)
            else:
                mean = self.R[i].mapper._mean + self.rule_vec[i]

                # Make sure mean stays in range
                if not 0 <= mean <= 1:
                    mean = np.clip(mean, 0, 1)
                    # Flip change direction
                    self.rule_vec[i] = -self.rule_vec[i]

            self._R[i] = RuleLeaf(self.R[i].feat,
                                  GaussianMapper(mean, self.std))

    def learn_and_record(self, features, opinions):
        """Update the different learning methods and recods statistics from
         them for analysis.

        :param features:
            Features of the artifact.
        :param opinions:
            Agent's neighbors' opinions about the artifact.
         """
        self.stats['opinions'].append(opinions)

        # Record max and random rewards
        best_eval = opinions[max(opinions, key=opinions.get)]
        self.stats['max_rewards'].append(best_eval)
        rnd_rewards = np.sum(list(opinions.values())) / len(opinions)
        self.stats['random_rewards'].append(rnd_rewards)

        if self.std is not None:
            # Non-linear stochastic gradient descent selection and update
            sgd_chosen_addr = self.learner.sgd_choose(features)
            sgd_reward = opinions[sgd_chosen_addr]
            self.learner.update_sgd(sgd_reward, sgd_chosen_addr, features)
            record_stats(self.stats, 'sgd', sgd_chosen_addr,
                         sgd_reward, sgd_reward == best_eval)

        # Q-learning selection and update
        bandit_chosen_addr = self.learner.bandit_choose()
        bandit_reward = opinions[bandit_chosen_addr]
        self.learner.update_bandit(bandit_reward, bandit_chosen_addr)
        record_stats(self.stats, 'bandit', bandit_chosen_addr,
                     bandit_reward, bandit_reward == best_eval)

        # Linear regression selection and update
        linear_chosen_addr = self.learner.linear_choose(features)
        linear_reward = opinions[linear_chosen_addr]
        self.learner.update_linear_regression(linear_reward, linear_chosen_addr, features)
        record_stats(self.stats, 'linear', linear_chosen_addr,
                     linear_reward, linear_reward == best_eval)

        # Polynomial regression selection and update
        poly_chosen_addr = self.learner.poly_choose(features)
        poly_reward = opinions[poly_chosen_addr]
        self.learner.update_poly(poly_reward, poly_chosen_addr, features)
        record_stats(self.stats, 'poly', poly_chosen_addr,
                     poly_reward, poly_reward == best_eval)

    async def gather_opinions(self, addrs, artifact):
        """Gather opinions about the artifact from agent addresses.
        """
        # Gather evaluations from the other agents
        opinions = {}
        for addr in addrs:
            remote_agent = await self.env.connect(addr)
            opinion, _ = await remote_agent.evaluate(artifact)
            opinions[addr] = opinion
        return opinions

    async def send_artifact(self, addr, artifact, eval, framings=None):
        remote_agent = await self.env.connect(addr)
        await remote_agent.give_artifact(artifact, eval, self.addr)

    @aiomas.expose
    def give_artifact(self, artifact, eval, addr):
        features = self.get_features(artifact)
        self.learner.update_sgd(eval, addr, features)
        self.learner.update_linear_regression(eval, addr, features)
        self.learner.update_bandit(eval, addr)


    def create_artifact(self):
        artifact = self.artifact_cls.invent(self.search_width, self, self.create_kwargs)[0][0]
        if type(artifact) == list:
            artifact = artifact[0][0]
        features = self.get_features(artifact)
        eval, _ = self.evaluate(artifact)
        if eval >= self._own_threshold:
            self.learn(artifact)
        return artifact, features, eval

    @aiomas.expose
    async def act(self):
        """If active, create an artifact and send it to everyone for evaluation.
        Update models based on the evaluation received from the connection
        chosen by the model.
        """
        self.age += 1

        non_active_send = np.random.rand() < self.send_prob

        # Create an artifact if agent is active or has memory
        if self.active or self.stmem.length > 0 or non_active_send:
            artifact, features, eval = self.create_artifact()

        # Stop here if not the active agent
        if not self.active:
            if non_active_send:
                # Send artifact to the active agent
                addr = list(self.connections)[0]
                await self.send_artifact(addr, artifact, eval)
            self.update_means()
            return

        opinions = await self.gather_opinions(self.connections.keys(), artifact)
        self.learn_and_record(features, opinions)

    def save_stats(self):
        # Save stats to a file
        if self.own_folder:
            path = os.path.join(self.data_folder, agent_name_parse(self.name))
        else:
            path = self.data_folder
        if not os.path.exists(path):
            os.mkdir(path)
        files_in_path = len(os.listdir(path))
        pickle_path = os.path.join(path, 'stats{}.p'.format(files_in_path + 1))
        pickle.dump(self.stats, open(pickle_path, 'wb'))

    @aiomas.expose
    def close(self, folder=None):
        if not self.active:
            return

        self.save_stats()

        # Log stats about the run
        max_reward = np.sum(self.stats['max_rewards'])
        random_reward = np.sum(self.stats['random_rewards'])
        linear_reward = np.sum(self.stats['linear']['rewards'])
        linear_chose_best = np.sum(self.stats['linear']['chose_best'])
        bandit_reward = np.sum(self.stats['bandit']['rewards'])
        bandit_chose_best = np.sum(self.stats['bandit']['chose_best'])
        poly_reward = np.sum(self.stats['poly']['rewards'])
        poly_chose_best = np.sum(self.stats['poly']['chose_best'])

        if self.std is not None:
            sgd_reward = np.sum(self.stats['sgd']['rewards'])
            sgd_chose_best = np.sum(self.stats['sgd']['chose_best'])
            self._log(logging.INFO, 'Non-linear SGD reward: {} ({}%), optimal {}/{} times'
                      .format(int(np.around(sgd_reward)),
                              int(np.around(sgd_reward / max_reward, 2) * 100),
                              sgd_chose_best, self.age))

        self._log(logging.INFO, 'Linear regression reward: {} ({}%), optimal {}/{} times'
                  .format(int(np.around(linear_reward)),
                          int(np.around(linear_reward / max_reward, 2) * 100),
                          linear_chose_best, self.age))
        self._log(logging.INFO, 'Polynomial regression reward: {} ({}%), optimal {}/{} times'
                  .format(int(np.around(poly_reward)),
                          int(np.around(poly_reward / max_reward, 2) * 100),
                          poly_chose_best, self.age))
        self._log(logging.INFO, 'Bandit reward: {} ({}%), optimal {}/{} times'
                  .format(int(np.around(bandit_reward)),
                          int(np.around(bandit_reward / max_reward, 2) * 100),
                          bandit_chose_best, self.age))
        self._log(logging.INFO, 'Max reward: ' + str(int(np.around(max_reward))))
        self._log(logging.INFO, 'Expected random reward: {} ({}%)'
                  .format(int(np.around(random_reward)),
                          int(np.around(random_reward / max_reward, 2) * 100)))
        self._log(logging.INFO, 'Connection totals:')

        for connection in self.connections.keys():
            total = 0
            for opinions in self.stats['opinions']:
                total += opinions[connection]
            self._log(logging.INFO, '{}: {}'.format(connection, int(np.around(total))))
        self._log(logging.INFO, 'Bandit perceived as best: {}'
                  .format(max(self.learner.bandits, key=self.learner.bandits.get)))


class SingleAgent(MultiAgent):
    """Agent that uses only one model. All choices are not simulated."""
    def __init__(self, environment, model, learn_from_received=False, *args, **kwargs):
        super().__init__(environment, *args, **kwargs)
        self.stats = {'rewards': [],
                      'random_rewards': [],
                      'critiques': [],
                      'chosen_neighbours': []}

        assert model in ['linear', 'sgd', 'Q'], 'No such model.'
        self.model = model
        self.learn_from_received = learn_from_received
        self.step_critiques = []

    def update_model(self, eval, addr, features=None):
        if self.model == 'sgd':
            self.learner.update_sgd(eval, addr, features)
        elif self.model == 'linear':
            self.learner.update_linear_regression(eval, addr, features)
        elif self.model == 'Q':
            self.learner.update_bandit(eval, addr)

    @aiomas.expose
    def give_artifact(self, artifact, eval, addr, random=False):
        """
        Another agent uses this function to give this agent an artifact.

        :param artifact:
            The artifact this agent receives.
        :param eval:
            Evaluation of the artifact by the giving agent.
        :param addr:
            Address of the giving agent.
        :return:
            Evaluation of the artifact by the receiving agent.
        """
        self_eval, _ = self.evaluate(artifact)
        features = self.get_features(artifact)
        if self.learn_from_received:
            if self.model == 'Q':
                self.update_model(self_eval, addr)
            else:
                self.update_model(eval, addr, features)
        self.step_critiques.append((addr, self_eval))
        return self_eval, None

    @aiomas.expose
    async def act(self):
        """Create an artifact and send it to a neighbour."""
        artifact, features, eval = self.create_artifact()

        # Choose neighbour
        neighbour_addr = {'sgd': self.learner.sgd_choose(features),
                          'linear': self.learner.linear_choose(features),
                          'Q': self.learner.bandit_choose()}[self.model]

        # Give artifact to neighbour and get opinion
        remote_agent = await self.env.connect(neighbour_addr)
        opinion, _ = await remote_agent.give_artifact(artifact, eval, self.addr)
        self.stats['rewards'].append(opinion)
        self.stats['chosen_neighbours'].append(neighbour_addr)

        # Choose a random neighbour
        random_addr = np.random.choice(list(self.connections.keys()))
        remote_agent = await self.env.connect(random_addr)
        random_opinion, _ = await remote_agent.evaluate(artifact)
        self.stats['random_rewards'].append(random_opinion)

        self.update_model(opinion, neighbour_addr, features)

    @aiomas.expose
    async def finalize_step(self):
        self.stats['critiques'].append(self.step_critiques)
        self.step_critiques = []

    @aiomas.expose
    def close(self, folder=None):
        self.save_stats()


class GPImageAgent(FeatureAgent):
    """Agent creating images with genetic programming.
    """
    def __init__(self, *args, **kwargs):
        save_folder = kwargs.pop('save_folder', None)
        cm_name = kwargs.pop('cm_name', None)
        self.output_shape = kwargs.pop('output_shape', (64, 64))
        super().__init__(*args, **kwargs)
        self.pset = kwargs['create_kwargs']['pset']

        self.save_id = 1
        if save_folder is not None:
            self.artifact_save_folder = os.path.join(save_folder,
                                                     self.sanitized_name())
            self._recreate_save_folder()
            self.color_map = self._create_color_map(cm_name)

    def _recreate_save_folder(self):
        import shutil
        if os.path.exists(self.artifact_save_folder):
            shutil.rmtree(self.artifact_save_folder)
        os.makedirs(self.artifact_save_folder)

    def _create_color_map(self, cm_name):
        from matplotlib import cm
        cm_name = 'viridis' if cm_name is None else cm_name
        x = np.linspace(0.0, 1.0, 256)
        return cm.get_cmap(cm_name)(x)[np.newaxis, :, :3][0]

    def save_artifact(self, a):
        if self.artifact_save_folder is None:
            return

        self._log(logging.INFO, "Saving artifact {} with output shape {}"
                  .format(self.save_id, self.output_shape))
        self.artifact_cls.save_artifact(a,
                                        self.artifact_save_folder,
                                        self.save_id,
                                        a.evals[self.name],
                                        self.pset,
                                        self.color_map,
                                        self.output_shape)
        self.save_id += 1

    @aiomas.expose
    def evaluate(self, artifact):
        """Evaluates an artifact based on value and novelty.
        """
        if self.name in artifact.evals:
            return artifact.evals[self.name], artifact.framings[self.name]

        evaluation = value = 0.0
        novelty = None

        # Test png image compression. If image is compressed to less that 8% of
        # the original (bmp image has 1078 bytes overhead in black & white
        # images), then the image is deemed too simple and evaluation is 0.0.
        if GIA.png_compression_ratio(artifact) >= 0.08:
            value, _ = super().evaluate(artifact)
            evaluation = value
            if self.novelty_weight != -1:
                novelty = self.novelty(artifact)
                evaluation = (1 - self.novelty_weight) * value + self.novelty_weight * novelty

        fr = {'value': value, 'novelty': novelty}
        artifact.add_eval(self, evaluation, fr)
        return evaluation, fr

    @aiomas.expose
    async def act(self):
        # Create and evaluate an artifact
        artifacts = self.invent(self.search_width, n_artifacts=1)
        for artifact, _ in artifacts:
            e, fr = self.evaluate(artifact)
            n = None if fr['novelty'] is None else np.around(fr['novelty'], 2)
            self._log(logging.INFO,
                      'Created an artifact with evaluation {} (v: {}, n: {})'
                      .format(np.around(e, 2), np.around(fr['value'], 2), n))
            self.add_artifact(artifact)

            if e >= self._own_threshold:
                self.learn(artifact)

            # Save artifact to save folder
            self.save_artifact(artifact)



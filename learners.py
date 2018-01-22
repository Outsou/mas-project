import numpy as np
from creamas.math import gaus_pdf
from operator import itemgetter
from scipy.stats import norm
import math


class MultiLearner():
    """A learner that is capable of using and updating multiple models.

    Currently implemented:
    stochastic gradient descent (SGD),
    Q-learning for n-armed bandit problem
    """

    def __init__(self, addrs, num_of_features, std=None,
                 centroid_rate=0.01, weight_rate=0.2, e=0.2, reg_weight=0.5, gauss_mem=10):
        """
        :param list addrs:
            Addresses of the agents that are modeled.
        :param int num_of_features:
            Number of features in an artifact.
        :param float std:
            Standard deviation for evaluating artifacts.
        :param float centroid_rate:
            Learning rate for centroids.
        :param float weight_rate:
            Learning rate for weights.
        :param int gauss_mem:
            How many latest artifacts the gaussian method remembers. If <= 0, the completely online update method
            is used.
        """
        self.centroid_rate = centroid_rate
        self.weight_rate = weight_rate
        self.num_of_features = num_of_features
        self.std = std
        self.e = e
        self.reg_weight = reg_weight
        self.gaussians = {}
        self.gauss_mem = gauss_mem

        # Get length of the polynomial transform
        poly_len = len(self._poly_transform(np.zeros(num_of_features)))

        # Initialize parameters
        self.poly_weights = {}
        self.linear_weights = {}
        self.sgd_weights = {}
        self.centroids = {}
        self.bandits = {}
        for addr in addrs:
            init_vals = [0.5] * num_of_features
            self.linear_weights[addr] = np.append(init_vals, init_vals[0])
            self.sgd_weights[addr] = np.array(init_vals)
            self.centroids[addr] = np.array(init_vals)
            self.bandits[addr] = 1
            self.poly_weights[addr] = np.array([0.5] * poly_len)
            self.gaussians[addr] = {'mean': 0.5, 'var': 0.08, 'vals': []}

        if std is not None:
            self.max = gaus_pdf(1, 1, std)

    @staticmethod
    def _poly_transform(features):
        length = len(features)
        transformation = []
        for i in range(length):
            transformation.append(features[i] ** 2)
            transformation.append(np.sqrt(2) * features[i])
        for i in range(length - 1):
            transformation.append(np.sqrt(2) * features[-1] * features[i])
        for i in range(1, length - 2):
            transformation.append(np.sqrt(2) * features[-2] * features[i])
        for i in range(1, length - 1):
            transformation.append(np.sqrt(2) * features[i] * features[0])
        transformation.append(1)
        return np.array(transformation)

    def _sgd_estimate(self, addr, features):
        """Estimate value the SGD model.
        Returns the estimate and individual values for different features.
        """
        assert self.std != None, 'std not set'

        vals = np.zeros(self.num_of_features)
        for i in range(self.num_of_features):
            vals[i] = gaus_pdf(features[i], self.centroids[addr][i], self.std) / self.max
        estimate = np.sum(self.sgd_weights[addr] * vals)
        return estimate, vals

    def update_bandit(self, true_value, addr, discount_factor=0,
                      learning_factor=0.9):
        """
        Updates the Q-value for addr.

        :param true_value:
            The evaluation of the artifact from addr.
        :param addr:
            The connection that will be updated.
        :param discount_factor:
            Controls how much the reward from future is discounted.
        :param learning_factor:
            Controls learning speed.
        """
        old_value = self.bandits[addr]
        self.bandits[addr] += learning_factor * (true_value + discount_factor * old_value - old_value)

    def update_sgd(self, true_value, addr, features):
        """
        Uses SGD to update weights and centroid for the SGD model.
        :param true_value:
            The evaluation of the artifact from addr.
        :param addr:
            The connection that will be updated.
        :param features:
            Objective features of an artifact.
        """
        assert self.std != None, 'std not set'

        estimate, vals = self._sgd_estimate(addr, features)

        error = true_value - estimate

        # Update weights
        regularizer = self.sgd_weights[addr] * 2 * self.reg_weight
        gradient = vals * error - regularizer
        self.sgd_weights[addr] += self.weight_rate * gradient
        self.sgd_weights[addr] = np.clip(self.sgd_weights[addr], 0, 1)

        # Calculate gradient of gaussian pdf w.r.t mean
        gradient = (features - self.centroids[addr]) \
                   * np.exp(-(features - self.centroids[addr]) ** 2 / (2 * self.std ** 2)) \
                   / (np.sqrt(2 * np.pi) * (self.std ** 2) ** (3 / 2))

        # Update centroid
        self.centroids[addr] += self.centroid_rate * gradient * error
        self.centroids[addr] = np.clip(self.centroids[addr], 0, 1)

    def update_linear_regression(self, true_value, addr, features):
        feature_vec = np.append(features, 1)
        error = true_value - np.sum(self.linear_weights[addr] * feature_vec)
        regularizer = self.linear_weights[addr] * 2 * self.reg_weight
        gradient = feature_vec * error - regularizer
        self.linear_weights[addr] += self.weight_rate * gradient

    def update_poly(self, true_value, addr, features):
        poly_feats = self._poly_transform(features)
        error = true_value - np.sum(self.poly_weights[addr] * poly_feats)
        gradient = poly_feats * error
        self.poly_weights[addr] += self.weight_rate * gradient


    def update_gaussian(self, val, addr):
        if self.gauss_mem > 0:
            self.gaussians[addr]['vals'].append(val)
            if len(self.gaussians[addr]['vals']) > self.gauss_mem:
                del self.gaussians[addr]['vals'][0]
            self.gaussians[addr]['mean'] = np.mean(self.gaussians[addr]['vals'])
            if len(self.gaussians[addr]['vals']) > 1:
                self.gaussians[addr]['var'] = np.var(self.gaussians[addr]['vals'])
        else:
            mean = self.gaussians[addr]['mean']
            var = self.gaussians[addr]['var']

            delta = val - mean
            self.gaussians[addr]['mean'] += delta * 0.1
            self.gaussians[addr]['var'] += (delta**2 - var) * 0.2


    def sgd_choose(self, features):
        """
        Choose a connection with the SGD model.

        :param features:
            Objective features of an artifact.
        :return:
            The chosen connection's address.
        """
        if np.random.rand() < self.e:
            return np.random.choice(list(self.sgd_weights.keys()))

        best_estimate = -1
        best_addr = None

        for addr in self.sgd_weights.keys():
            estimate, _ = self._sgd_estimate(addr, features)
            if estimate > best_estimate:
                best_estimate = estimate
                best_addr = addr

        return best_addr

    def poly_choose(self, features):
        if np.random.rand() < self.e:
            return np.random.choice(list(self.poly_weights.keys()))

        poly_feats = self._poly_transform(features)
        best_estimate = -np.inf
        best_addr = None

        for addr in self.poly_weights.keys():
            estimate = np.sum(self.poly_weights[addr] * poly_feats)
            if estimate > best_estimate:
                best_estimate = estimate
                best_addr = addr

        return best_addr

    def bandit_choose(self, get_list=False):
        """Choose a connection with the Q-learning model.

        :return:
            The chosen connection's address.
        """
        if np.random.rand() < self.e:
            if get_list:
                keys = list(self.bandits.keys())
                np.random.shuffle(keys)
                return keys
            return np.random.choice(list(self.bandits.keys()))

        if get_list:
            keys, vals = zip(*sorted(self.bandits.items(), key=itemgetter(1), reverse=True))
            return keys

        best = -np.inf
        best_addr = None

        for addr, value in self.bandits.items():
            if value > best:
                best = value
                best_addr = addr

        return best_addr

    def linear_choose(self, features):
        """Choose a connection with the linear regression model.

        :param features:
            List of features of the artifact.
        :return:
            The chosen connection's address.
        """
        if np.random.random() < self.e:
            return np.random.choice(list(self.linear_weights.keys()))

        addrs = list(self.linear_weights.keys())
        feature_vec = np.append(features, 1)
        best_estimate = np.sum(self.linear_weights[addrs[0]] * feature_vec)
        best_addr = addrs[0]

        for addr in addrs[1:]:
            estimate = np.sum(self.linear_weights[addr] * feature_vec)
            if estimate > best_estimate:
                best_estimate = estimate
                best_addr = addr

        return best_addr

    def linear_choose_multi(self, artifacts):
        """Returns a list of addresses in order of highest to lowest average evaluation of the artifacts.

        :param artifacts:
            List containing feature lists of the artifacts.
        :return:
            List of addresses.
        """
        addrs = list(self.linear_weights.keys())

        if np.random.random() < self.e:
            np.random.shuffle(addrs)
            return addrs

        estimate_dict = {}
        for addr in addrs:
            estimate_dict[addr] = 0

        for artifact in artifacts:
            feature_vec = np.append(artifact, 1)
            for addr in estimate_dict.keys():
                estimate_dict[addr] += np.sum(self.linear_weights[addr] * feature_vec)

        sorted_addrs, _ = zip(*sorted(estimate_dict.items(), key=itemgetter(1), reverse=True))
        return sorted_addrs


    def gaussian_choose(self, target, get_list=False):
        def get_mean_dist(mean, var, point):
            diff_mean = mean - point
            dist_mean = np.sqrt(var) * np.sqrt(2 / np.pi) * np.exp(-diff_mean ** 2 / (2 * var)) \
                        - diff_mean * math.erf(-diff_mean / np.sqrt(2 * var))
            return dist_mean

        if np.random.rand() < self.e:
            if get_list:
                keys = list(self.gaussians.keys())
                np.random.shuffle(keys)
                return keys
            return np.random.choice(list(self.gaussians.keys()))

        dist_dict = {}
        for addr, gauss in self.gaussians.items():
            # prob = norm.cdf(target + 0.1, gauss['mean'], np.sqrt(gauss['var'])) \
            #        - norm.cdf(target - 0.1, gauss['mean'], np.sqrt(gauss['var']))

            # prob =  norm.pdf(target, gauss['mean'], np.sqrt(gauss['var'])) / norm.pdf(0, 0, np.sqrt(gauss['var']))

            # Math from https://en.wikipedia.org/wiki/Folded_normal_distribution
            diff_mean = gauss['mean'] - target
            dist_mean = np.sqrt(gauss['var']) * np.sqrt(2 / np.pi) * np.exp(-diff_mean ** 2 / (2 * gauss['var'])) \
                        - diff_mean * math.erf(-diff_mean / np.sqrt(2 * gauss['var']))

            dist_dict[addr] = dist_mean


        keys, vals = zip(*sorted(dist_dict.items(), key=itemgetter(1)))

        if get_list:
            return keys
        else:
            return keys[0]

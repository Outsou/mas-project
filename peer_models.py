from operator import itemgetter
import numpy as np


class QLearner:
    def __init__(self, init_val, state_action_dict, learning_rate=0.9, discount_factor=0.9, e=0.2):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.e = e
        self.q_vals = {}
        for state, actions in state_action_dict.items():
            self.q_vals[state] = {}
            for action in actions:
                self.q_vals[state][action] = init_val

    def update(self, reward, state, action):
        old_val = self.q_vals[state][action]
        self.q_vals[state][action] += self.learning_rate * (reward + self.discount_factor * old_val - old_val)

    def decide_action(self, state, get_list=False):
        if np.random.rand() < self.e:
            actions = list(self.q_vals[state].keys())
            return np.random.permutation(actions) if get_list else np.random.choice(actions)
        if get_list:
            sorted_actions, _ = zip(*sorted(self.q_vals[state], key=itemgetter(1), reverse=True))
            return sorted_actions
        return max(self.q_vals[state].items(), key=itemgetter(1))[0]

class BanditPeerModel:
    def __init__(self, init_val=0.8, learning_rate=0.9, e=0):
        self.init_val = init_val
        # Initialize QLearner with one state and no actions
        self.learner = QLearner(init_val, {0: []}, learning_rate, 0, e)

    def update(self, value, peer):
        self.learner.update(value, 0, peer)

    def choose_peer(self, get_list=False):
        return self.learner.decide_action(0, get_list)

    def add_peer(self, peer):
        self.learner.q_vals[0][peer] = self.init_val

    def remove_peer(self, peer):
        self.learner.q_vals[0].pop(peer, None)

class BinPeerModel:
    def __init__(self, num_of_bins, lower_bound, upper_bound, init_val=0.8, learning_rate=0.9, e=0):
        self.init_val = init_val
        self.num_of_bins = num_of_bins
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        # Initialize QLearner with a state for each bin with no actions
        state_action_dict = {}
        for i in range(self.num_of_bins):
            state_action_dict[i] = []
        self.learner = QLearner(init_val, state_action_dict, learning_rate, 0, e)

        self.bin_size = (self.upper_bound - self.lower_bound) / self.num_of_bins
        self.bin_goals = []
        for i in range(self.num_of_bins):
            start = self.lower_bound + i * self.bin_size
            end = start + self.bin_size
            middle = (start + end) / 2
            self.bin_goals.append(middle)

    def bin_to_goal(self, bin):
        return self.bin_goals[bin]

    def goal_to_bin(self, goal):
        dists = [abs(x - goal) for x in self.bin_goals]
        return np.argmin(dists)

    def update(self, value, peer, goal):
        bin = self.goal_to_bin(goal)
        self.learner.update(value, bin, peer)

    def choose_peer(self, goal, get_list=False):
        bin = self.goal_to_bin(goal)
        return self.learner.decide_action(bin, get_list)

    def add_peer(self, peer):
        for action_dict in self.learner.q_vals.values():
            action_dict[peer] = self.init_val

    def remove_peer(self, peer):
        for action_dict in self.learner.q_vals.values():
            action_dict.pop(peer, None)

    def get_best_bin(self, peer):
        bin_vals = {}
        for bin, vals in self.learner.q_vals.items():
            bin_vals[bin] = vals[peer]
        return max(bin_vals.items(), key=itemgetter(1))[0]

    def get_best_top_n_bin(self, bins, n):
        """
        Returns the bins in order of sum for top n addrs.
        :param bins:
            List of bins from which the best bin is calculated.
        :param n:
            The top n q_values for each bin that are summed.
        :return:
            The bins and their top n q-value sums in order.
        """
        bin_vals = {}
        for bin in bins:
            vals = sorted(self.learner.q_vals[bin].values(), reverse=True)
            bin_vals[bin] = sum(vals[:n])
        bins_sorted = sorted(bin_vals.items(), key=itemgetter(1), reverse=True)
        return bins_sorted

    def filter_bins(self):
        """
        Returns a set of states where no addr has its highest value and a set of the states where some addr
        has its highest value.
        """
        occupied = set()
        for peer in self.learner.q_vals[0].keys():
            vals = [(bin, self.learner.q_vals[bin][peer]) for bin in self.learner.q_vals.keys()]
            best_bin = max(vals, key=itemgetter(1))[0]
            occupied.add(best_bin)
        free = set(self.learner.q_vals.keys()) - occupied
        return free, occupied

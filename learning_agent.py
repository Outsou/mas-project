from creamas import CreativeAgent
import numpy as np
from operator import itemgetter


class LearningAgent(CreativeAgent):
    def __init__(self, num_of_bins=1, init_val=0.8, learning_rate=0.9, goal_attribute=None, aesthetic_bounds=None):
        self.num_of_bins = num_of_bins
        self.init_val = init_val
        self.learning_rate = learning_rate
        self.goal_attribute = goal_attribute
        self.q_vals = []
        self.aesthetic_bounds = aesthetic_bounds
        for _ in range(self.num_of_bins):
            self.q_vals.append({})
        if self.goal_attribute is not None:
            assert self.aesthetic_bounds is not None, 'Bounds not given for goal attribute.'
            self.bin_size = (self.aesthetic_bounds[1] - self.aesthetic_bounds[0]) / self.num_of_bins
            self.bin_mids = []
            self.bin_borders = []
            for i in range(self.num_of_bins):
                start = self.aesthetic_bounds[0] + i * self.bin_size
                self.bin_borders.append(start)
                end = start + self.bin_size
                middle = (start + end) / 2
                self.bin_mids.append(middle)
            self.bin_borders.append(self.aesthetic_bounds[1])

    def get_current_bin(self):
        if self.num_of_bins == 1 or self.goal_attribute is None:
            return 0
        dists = [abs(x - getattr(self, self.goal_attribute)) for x in self.bin_mids]
        return np.argmin(dists)

    def update(self, artifact):
        e, fr = self.evaluate(artifact)
        if self.num_of_bins == 1 or self.goal_attribute is None:
            old_value = self.q_vals[0][artifact.creator]
            self.q_vals[0][artifact.creator] += self.learning_rate * (e - old_value)
        else:
            for i in range(len(self.bin_mids)):
                old_value = self.q_vals[i][artifact.creator]
                #TODO: eval
                eval = 0
                self.q_vals[i][artifact.creator] += self.learning_rate * (eval - old_value)

    def decide(self, get_list=True):
        bin = self.get_current_bin()
        vals = self.q_vals[bin]
        sorted_peers, _ = zip(*sorted(vals.items(), key=itemgetter(1), reverse=True))
        return sorted_peers if get_list else sorted_peers[0]

    def add_peer(self, addr):
        for bin in self.q_vals.keys():
            self.q_vals[bin][addr] = self.init_val

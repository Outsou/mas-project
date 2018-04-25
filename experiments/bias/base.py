"""
Implementations for collaboration subclasses.
"""
import logging
import time
import random
import operator
import os
import pickle
import asyncio
import pprint
import traceback

import numpy as np
from deap import tools, gp, creator, base
import aiomas
from creamas import Simulation, Artifact
from creamas.util import create_tasks, run, run_or_coro
from creamas.rules import RuleLeaf

from environments import StatEnvironment

from agents import GPImageAgent, FeatureAgent, agent_name_parse
from artifacts import GeneticImageArtifact as GIA
from experiments.collab.ranking import choose_best
from learners import MultiLearner
from features import ImageEntropyFeature, ImageComplexityFeature
from mappers import LinearDiffMapper


__all__ = ['BiasGPAgent',
           'BiasEnvironment',
           'BiasSimulation']


def csplit(creator):
    """Split creator name.

    :returns: tuple of creators. (length 1 if individual, length 2 if collab)
    """
    return creator.split(" - ")


def get_aid(addr, age, aest, val, nov, caddr=None, caest=None):
    aid = ""
    if caddr is None:
        aid = "{:0>5}{}_{}_v{}_n{}".format(
            age, agent_name_parse(addr), aest, val, nov)
    else:
        aid = "{:0>5}{}_{}_v{}_n{}-{}_{}".format(
            age, agent_name_parse(addr), aest, val, nov,
            agent_name_parse(caddr), caest)
    return aid


class ImageSTMemory:
    """Agent's short-term memory model using a simple list which stores
    artifacts as is.

    Right now designed to be used with :class:`GeneticImageArtifact`.
    """

    def __init__(self, artifact_cls, max_dist, max_length=100):
        self.artifacts = []
        self.max_length = max_length
        self.artifact_cls = artifact_cls
        self.max_distance = max_dist

    def _add_artifact(self, artifact):
        # Lets add artifacts to the end of the list and remove the start of the
        # list when it exceeds twice the memory length. Insertion to the start
        # of the list takes more time (iirc) than appending to the end.
        if len(self.artifacts) >= 2 * self.max_length:
            self.artifacts = self.artifacts[-self.max_length:]
        self.artifacts.append(artifact)

    def get_artifacts(self, creator=None):
        """Get artifacts in the memory.

        If ``creator`` is not None, returns only artifacts which have
        that creator. It should be the name of the creating agent.
        (Or the collaboration addresses in the form of `myaddr1 - otheraddr2`)
        """
        arts = self.artifacts[-self.max_length:]
        if creator is None:
            return arts

        filtered_arts = []  # f_arts kjehkjeh
        for a in arts:
            if a.creator == creator:
                filtered_arts.append(a)
        return filtered_arts

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
        """Return distance to the closest artifact in the memory (normalized to
        in [0, 1]), or 1.0 if memory is empty.
        """
        mem_artifacts = self.get_artifacts()
        if len(mem_artifacts) == 0:
            return 1.0

        min_distance = self.max_distance
        for a in mem_artifacts:
            d = self.artifact_cls.distance(artifact, a)
            if d < min_distance:
                min_distance = d
        return min_distance / self.max_distance


class BiasGPAgent(GPImageAgent):
    """GP Agent which has a bias.
    """
    def __init__(self, *args, **kwargs):
        self.pset_names = kwargs.pop('pset_names', [])
        self.aesthetic = kwargs.pop('aesthetic', '')
        self.gender = 1 if random.random() <= kwargs.pop('gender_balance', 0.5) else 0
        self.has_bias = 1 if random.random <= kwargs.pop('bias_prob', 0.0) else 0
        bias_dist = kwargs.pop('bias_dist', (0.0, 0.0))
        self.max_bias_value = 0.0000000000001
        if self.has_bias:
            # How much worse other gender's artifacts are evaluated
            self.bias = random.random() * bias_dist[1] + bias_dist[0]
            self.bias = max((min([0.0, self.bias]), 1.0))
        else:
            self.bias = 0.0

        super().__init__(*args, **kwargs)
        self.pop_size = self.create_kwargs['pop_size']
        md = GIA.max_distance(self.create_kwargs)
        self.stmem = ImageSTMemory(GIA, md, self.mem_size)
        self.age = 0
        self.last_artifact = None
        # AIDs of all seen artifacts.
        self.seen_artifacts = []

        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("SuperIndividual",
                       gp.PrimitiveTree,
                       fitness=creator.FitnessMax,
                       pset=self.super_pset,
                       image=None)

        self.creator_name = "Individual_{}".format(self.sanitized_name())
        creator.create(self.creator_name,
                       gp.PrimitiveTree,
                       fitness=creator.FitnessMax,
                       pset=self.pset,
                       image=None)

        # Data structures to keep information about the created artifacts.
        # Mainly evaluations, e.g. value and novelty for own artifacts,
        # but also for collaborated artifacts the partner's aesthetic function.
        self.own_arts = {'val': [],             # Own value (non-norm.)
                         'nov': [],             # Own novelty
                         'eval': [],            # Own overall evaluation
                         'aid': [],             # Image ID
                         'mval': [],            # Current max own value
                         'nval': [],            # Normalized own value
                         'neval': [],           # Normalized own evaluation
                         'age': []}             # Own age

        # Evaluations of own artifacts by all the agents: key is aid
        # values are dicts
        self.own_evals = {}
        self.save_general_info()

    def get_features(self, artifact):
        """Return objective values for features without mapping.
        """
        features = []
        for rule in self.R:
            features.append(rule.feat(artifact))
        return features

    @aiomas.expose
    def get_aesthetic(self):
        return self.aesthetic

    def save_general_info(self):
        sfold = self.artifact_save_folder
        info = {'aesthetic': self.aesthetic,
                'pset_names': self.pset_names,
                'addr': self.addr,
                'gender': self.gender,
                'has_bias': self.has_bias,
                'bias': self.bias,
                }

        with open(os.path.join(sfold, 'general_info.pkl'), 'wb') as f:
            pickle.dump(info, f)

    def append_oa(self, artifact):
        fr = artifact.framings[self.name]
        self.own_arts['val'].append(fr['value'])
        self.own_arts['nov'].append(fr['novelty'])
        self.own_arts['eval'].append(artifact.evals[self.name])
        self.own_arts['aid'].append(artifact.aid)
        self.own_arts['mval'].append(fr['max_value'])
        self.own_arts['nval'].append(fr['norm_value'])
        self.own_arts['neval'].append(fr['norm_evaluation'])
        self.own_arts['age'].append(self.age)

    def add_own_evals(self, aid, evals):
        self.own_evals[aid] = {}
        for addr, e in evals:
            self.own_evals[aid][addr] = e

    @aiomas.expose
    def get_arts_information(self):
        return self.aesthetic, self.own_arts

    @aiomas.expose
    def evaluate(self, artifact, use_png_compression=True):
        """Evaluates an artifact based on value (and novelty).

        ADDS BIAS TO THE EVALUATION.

        :param bool use_png_compression:
            If ``True`` checks artifact's compression ratio with PNG. If
            resulting image is too small (compresses too much w.r.t. original),
            gives evaluation 0.0 for the image.
        """
        if self.name in artifact.evals:
            return artifact.evals[self.name], artifact.framings[self.name]

        evaluation = value = 0.0
        novelty = None

        # Test png image compression. If image is compressed to less that 8% of
        # the original (bmp image has 1078 bytes overhead in black & white
        # images), then the image is deemed too simple and evaluation is 0.0.
        if use_png_compression and not artifact.png_compression_done:
            png_ratio = GIA.png_compression_ratio(artifact)
            artifact.png_compression_done = True
            if png_ratio < 0.08:
                fr = {'value': value,
                      'novelty': 0.0,
                      'pass_novelty': False,
                      'pass_value': False,
                      'max_value': self.max_value,
                      'norm_value': 0.0,
                      'norm_evaluation': 0.0,
                      'aesthetic': self.aesthetic,
                      'max_bias_value': self.max_bias_value,
                      'bias_value': 0.0,
                      'has_bias': self.has_bias,
                      'bias_evaluation': 0.0,
                      'norm_bias_value': 0.0,
                      'norm_bias_evaluation': 0.0,
                      }
                artifact.add_eval(self, evaluation, fr)
                return evaluation, fr

        # Call RuleAgent's evaluate, not FeatureAgent's!
        # Otherwise FeatureAgent will include novelty into the evaluation if
        # the novelty weight != -1. This will then affect the observed value
        # and mess up the value-based predictions.
        value, _ = super(FeatureAgent, self).evaluate(artifact)

        # ADD BIAS
        if self.has_bias and artifact.creator_gender != self.gender:
            bias_value = (1 - self.bias) * value
        else:
            bias_value = value

        if self.max_bias_value < bias_value:
            self.max_bias_value = bias_value

        if self.max_value < value:
            self.max_value = value

        evaluation = value
        bias_evaluation = bias_value

        norm_value = value / self.max_value
        norm_bias_value = bias_value / self.max_bias_value
        normalized_evaluation = norm_value
        normalized_bias_evaluation = norm_bias_value

        if self.novelty_weight != -1:
            novelty = float(self.novelty(artifact))
            evaluation = (1.0 - self.novelty_weight) * value + self.novelty_weight * novelty
            normalized_evaluation = (1.0 - self.novelty_weight) * norm_value + self.novelty_weight * novelty
            bias_evaluation = (1.0 - self.novelty_weight) * bias_value + self.novelty_weight * novelty
            normalized_bias_evaluation = (1.0 - self.novelty_weight) * norm_bias_value + self.novelty_weight * novelty

        fr = {'value': value,
              'novelty': novelty,
              'pass_novelty': bool(novelty >= self._novelty_threshold) if novelty is not None else False,
              'pass_value': bool(value >= self._value_threshold),
              'max_value': self.max_value,
              'norm_value': norm_value,
              'norm_evaluation': normalized_evaluation,
              'aesthetic': self.aesthetic,
              'max_bias_value': self.max_bias_value,
              'has_bias': self.has_bias,
              'bias_value': bias_value,
              'bias_evaluation': bias_evaluation,
              'norm_bias_value': norm_bias_value,
              'norm_bias_evaluation': normalized_bias_evaluation,
              }
        artifact.add_eval(self, bias_evaluation, fr)
        return bias_evaluation, fr

    @aiomas.expose
    async def act_individual(self, *args, **kwargs):
        """Agent's act for individually made artifacts.
        """
        artifacts = self.invent(self.search_width, n_artifacts=1)
        for artifact, _ in artifacts:
            e, fr = self.evaluate(artifact)
            n = None if fr['novelty'] is None else np.around(fr['novelty'], 2)
            v = np.around(fr['norm_value'], 2)
            e = fr['norm_evaluation']
            self._log(logging.INFO,
                      '{} Created an individual artifact. E:{} (norm.), V:{} (norm.), N:{}'
                      .format(self.aesthetic.upper(), np.around(e, 2), v, n))

            aid = get_aid(self.addr, self.age, self.aesthetic, v, n)
            artifact.aid = aid
            artifact.creator_gender = self.gender
            # self.save_artifact(artifact, pset=self.super_pset, aid=aid)
            # Append to own artifacts
            self.append_oa(artifact)
            self.seen_artifacts.append(aid)

            # Artifact is acceptable if it passes value and novelty thresholds.
            # It is questionable if we need these, however.
            #passed = fr['pass_value'] and fr['pass_novelty']
            passed = fr['norm_value'] > self._value_threshold
            if passed:
                self._log(logging.DEBUG,
                          "Individual artifact passed thresholds")
                self.learn(artifact)
                self.last_artifact = artifact

    @aiomas.expose
    async def act(self, *args, **kwargs):
        self.last_artifact = None
        self.age += 1
        return await self.act_individual(*args, **kwargs)

    @aiomas.expose
    async def send_artifact(self, as_coro=True):
        """Send own last artifact (if not None) to connected peers.
        """
        async def task(addr, artifact):
            remote_agent = await self.connect(addr)
            ret = await remote_agent.show_artifact(artifact)
            return ret

        if self.last_artifact is not None:
            self._log(logging.DEBUG, "Sending artifact to peers")
            addrs = [addr for addr in self.connections.keys()]
            tasks = create_tasks(task, addrs, self.last_artifact)
            if as_coro:
                rets = await run_or_coro(tasks, as_coro)
            else:
                rets = run(tasks)
            self._log(logging.INFO, "Got response to artifact of length: {}".format(len(rets)))
            return rets

        return None

    @aiomas.expose
    async def show_artifact(self, artifact, as_coro=True):
        """Show artifact to others if seen first time and evaluated good enough.
        """
        async def task(addr, artifact):
            remote_agent = await self.connect(addr)
            ret = await remote_agent.show_artifact(artifact)
            return ret

        self._log(logging.DEBUG, "Looking at artifact created by {}".format(artifact.creator))
        # Do not learn artifact again if it has been shown/seen already.
        if artifact.aid not in self.seen_artifacts:
            self.seen_artifacts.append(artifact.aid)
        else:
            return
        if self.name in artifact.framings:
            # Already evaluated artifact (seen it, do nothing)
            return

        e, fr = self.evaluate(artifact)
        rets = []

        if fr['novelty'] > 0.4 and fr['norm_bias_value'] > 0.5:
            self.learn(artifact)
            # Show artifact to all peers which have not evaluated the artifact yet.
            addrs = [addr for addr in self.connections.keys() if addr not in artifact.framings]
            self._log(logging.DEBUG, "Sending artifact to: {}".format(addrs))
            tasks = create_tasks(task, addrs, artifact)
            if as_coro:
                rets = await run_or_coro(tasks, as_coro)
            else:
                rets = run(tasks)

        ret = [(self.addr, e)] + [r for r in rets if r is not None]
        return ret

    @aiomas.expose
    def rcv_evaluations_from_artifact(self, artifact, evaluations):
        """Receive evaluations (done by peers) from an own artifact.

        :param artifact: Artifact the evaluations belong to

        :param dict evaluations:
           Keys are addresses and values are evaluation, framing pairs.
        """
        self._log(logging.DEBUG,
                  "Got evals from {}".format(artifact.aid))
        return


    @aiomas.expose
    def get_last_artifact(self):
        return self.last_artifact

    @aiomas.expose
    def save_arts_info(self):
        sfold = self.artifact_save_folder

        with open(os.path.join(sfold, 'own_arts.pkl'), 'wb') as f:
            pickle.dump(self.own_arts, f)

        return self.aesthetic, self.own_arts, self.own_evals,


class BiasSimulation(Simulation):
    """An implementation of :class:`creamas.core.simulation.Simulation`
    subclass.

    The class allows for a pre-callback before every simulation step
    in order for agents to choose their collaboration partners (or environment
    to choose them for them).
    
    (This is largely redundant because of (post)callback allowed for a basic
    simulation run, but makes more sense conceptually.)
    """
    def __init__(self, env, callback=None, precallback=None, log_folder=None):
        super().__init__(env, callback, log_folder)
        self._precallback = precallback

    def _init_step(self):
        """Initialize next step of simulation to be run.
        """
        super()._init_step()
        if self._precallback is not None:
            self._precallback(self.age)


class BiasEnvironment(StatEnvironment):
    """Collaboration environment which matches collaboration partners based
    on each agents preferences and random rankings generated each time step.

    Random rankings are used to decide in which order the agents can choose
    their collaboration partners. Agent can belong to only one collaboration
    at a time.
    """

    def __init__(self, *args, **kwargs):
        self.save_folder = kwargs.pop('save_folder', None)
        super().__init__(*args, **kwargs)
        self.ind_evals = {}
        self.age = 0

    def analyse_arts_inform(self, agent, aest, own_arts):
        loa = len(own_arts['eval'])
        meval = 0
        mnovelty = 0
        mvalue = 0
        if loa > 0:
            meval = np.mean(own_arts['eval'])
            mnovelty = np.mean(own_arts['nov'])
            mvalue = np.mean(own_arts['val'])

        self._log(logging.INFO, "{} {:<10}: arts={} e={:.3f} n={:.3f} v={:.3f}"
                  .format(agent, aest.upper(), loa, meval, mnovelty, mvalue,))

    def analyse_all(self):
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            aest, oa = await r_agent.get_arts_information()
            return addr, aest, oa

        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        rets = run(tasks)
        for agent, aest, oa in rets:
            self.analyse_arts_inform(agent, aest, oa)

    def post_cbk(self, *args, **kwargs):
        self.disperse_artifacts()
        self.analyse_all()

    def disperse_artifacts(self):
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            ret = await r_agent.send_artifact()
            return addr, ret

        eval_dict = self.ind_evals
        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        rets = run(tasks)
        #print(rets)

    def collect_evaluations(self):
        """Collect evaluations from all agents for all artifacts made in the
        previous step.

        Send the evaluations to their creators afterwards.
        """
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            ret = await r_agent.get_last_artifact()
            return addr, ret

        async def slave_task2(addr, artifact):
            r_agent = await self.connect(addr, timeout=5)
            ret = await r_agent.show_artifact(artifact)
            return addr, ret

        async def slave_task3(addr, artifact, evaluations):
            r_agent = await self.connect(addr, timeout=5)
            ret = await r_agent.rcv_evaluations_from_artifact(artifact, evaluations)
            return addr, ret

        eval_dict = self.ind_evals

        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        rets = run(tasks)
        random.shuffle(rets)

        eval_tasks = []
        for r in rets:
            if r[1] is not None:
                tasks = create_tasks(slave_task2, addrs, r[1], flatten=False)
                rets2 = run(tasks)
                d = {ret[0]: ret[1] for ret in rets2}
                eval_dict[r[1].aid] = d
                eval_dict[r[1].aid]['creator'] = r[1].creator
                for addr in csplit(r[1].creator):
                    t = asyncio.ensure_future(slave_task3(addr, r[1], d))
                    eval_tasks.append(t)
        ret = run(asyncio.gather(*eval_tasks))
        return

    def save_artifact_info(self):
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            ret = await r_agent.save_arts_info()
            return addr, ret

        self._log(logging.INFO,
                  "Saving all info.")

        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        rets = run(tasks)
        sfold = self.save_folder

        with open(os.path.join(sfold, 'ind_evals.pkl'), 'wb') as f:
            pickle.dump(self.ind_evals, f)

        return rets, self.ind_evals

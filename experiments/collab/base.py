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

import numpy as np
from deap import tools, gp, creator, base
import aiomas
from creamas import Simulation
from creamas.util import create_tasks, run

from environments import StatEnvironment

from agents import GPImageAgent, agent_name_parse
from artifacts import GeneticImageArtifact as GIA
from experiments.collab.ranking import choose_best
from learners import MultiLearner

__all__ = ['CollaborationBaseAgent',
           'GPCollaborationAgent',
           'CollabEnvironment',
           'CollabSimulation']


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
        #print("artifacts in memory: {} ({})".format(
        #    len(self.get_artifacts()), len(self.artifacts)))

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


class CollaborationBaseAgent(GPImageAgent):
    """Base agent for collaborations, does not do anything in its :meth:`act`.
    """
    def __init__(self, *args, **kwargs):
        # Learning model by which collaboration partners are chosen
        self.collab_model = kwargs.pop('collab_model', 'random')
        self.collab_iters = kwargs.pop('collab_iters', 10)
        super().__init__(*args, **kwargs)
        self.in_collab = False  # Is the agent currently in collaboration
        self.caddr = None       # Address of the collaboration agent if any
        self.cinit = False      # Is this agent the initiator of collaboration
        md = GIA.max_distance(self.create_kwargs)
        self.stmem = ImageSTMemory(GIA, md, self.mem_size)
        self.learner = None

    @aiomas.expose
    def add_connections(self, conns):
        rets = super().add_connections(conns)

        # Initialize the multi-model learner
        self.learner = MultiLearner(list(self.connections),
                                    len(self.R))

        return rets

    @aiomas.expose
    def force_collab(self, addr, init):
        """Force agent to collaborate with an agent in given address.

        Used by :meth:`CollabEnvironment.match_collab_partners`.

        :param addr: Agent to collaborate with
        :param bool init: Is this agent the initiator in the collaboration.
        """
        self.in_collab = True
        self.caddr = addr
        self.cinit = init

    @aiomas.expose
    async def agree_on_collab(self, addr, rank):
        """Agree to collaborate with 'addr' if not already in collaboration.
        """
        if self.in_collab:
            print("{} already in collab, declining request from {}."
                  .format(self.addr, addr))
            return False
        else:
            if self.collab_rank > rank:
                print("{} accepting collaboration from {}."
                      .format(self.addr, addr))
                self.in_collab = True
                self.caddr = addr
                self.cinit = False
                return True

    @aiomas.expose
    async def get_collab_prefs(self):
        """Return a list of possible collaboration partners sorted in their
        preference order.

        This list is used to choose collaboration partners in
        :meth:`CollabEnvironment.match_collab_partners`.
        """
        if self.collab_model == 'random':
            partners = list(self.connections.keys())
            random.shuffle(partners)
        return self.addr, partners

    @aiomas.expose
    def clear_collab(self):
        """Clear current collaboration information.

        Used by :meth:`CollabEnvironment.clear_collabs`
        """
        self.in_collab = False
        self.caddr = None
        self.cinit = False

    async def find_collab_partner(self, method='random'):
        """Find collaboration partner from available connections.

        NOT USED IN THE ACTUAL COLLABORATIONS.

        See :class:`CollabEnvironment.match_collab_partners`
        """
        partner_found = False
        avail_partners = list(self.connections.keys())
        print(self.addr, self.connections.keys())
        while not partner_found and len(avail_partners) > 0:
            if self.in_collab:
                return
            addr = random.choice(avail_partners)
            print("{} asking collab from {}.".format(self.addr, addr))
            r_agent = await self.connect(addr)
            resp = await r_agent.agree_on_collab(self.addr, self.collab_rank)
            if resp and not self.in_collab:
                print("{} collaborates with {}."
                      .format(self.addr, addr))
                self.in_collab = True
                self.caddr = addr
                self.cinit = True
            partner_found = resp
            avail_partners.remove(addr)

    @aiomas.expose
    async def act(self):
        if not self.in_collab:
            print("{}: {} {} {}"
                  .format(self.addr, self.in_collab, self.caddr, self.cinit))


class GPCollaborationAgent(CollaborationBaseAgent):
    """Collaboration agent for GP images.
    """
    def __init__(self, *args, **kwargs):
        self.pset_names = kwargs.pop('pset_names', [])
        self.aesthetic = kwargs.pop('aesthetic', '')
        super().__init__(*args, **kwargs)
        self.pop_size = self.create_kwargs['pop_size']
        self.collab_hof_size = 20
        self.collab_hof = tools.HallOfFame(self.collab_hof_size)
        self.collab_pop = None
        self.age = 0
        self.last_artifact = None

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

        self.collab_arts = {'caddr': [],        # Collaborator's addr
                            'cinit': [],        # Initiator?
                            'fb': [],           # Found artifact?
                            'caest': [],        # Collab. aesthetic
                            'aid': [],          # Image ID
                            'val': [],          # Own value
                            'mval': [],         # Current max val
                            'nval': [],         # Normalized own value
                            'neval': [],        # Normalized own evaluation
                            'nov': [],          # Own novelty
                            'eval': [],         # Own overall evaluation
                            'cval': [],         # Collab's value (non-norm.)
                            'cnov': [],         # Collab's novelty
                            'ceval': [],        # Collab's overall evaluation
                            'cmval': [],        # Collab's current max val
                            'cnval': [],        # Collab's normalized val
                            'cneval': [],       # Collab's normalized evaluation
                            'age': []}          # Own age

        # Evaluations of collaborated artifacts by all the agents: key is aid
        # values are dicts
        self.collab_evals = {}
        self.save_general_info()

    def get_features(self, artifact):
        """Return objective values for features without mapping.
        """
        features = []
        for rule in self.R:
            features.append(rule.feat(artifact))
        return features

    @aiomas.expose
    async def get_collab_prefs(self):
        """Return a list of possible collaboration partners sorted in their
        preference order.

        This list is used to choose collaboration partners in
        :meth:`CollabEnvironment.match_collab_partners`.
        """
        self.init_collab()
        if self.collab_model == 'random':
            partners = list(self.connections.keys())
            random.shuffle(partners)

        if self.collab_model in ['Q0', 'Q1', 'Q2' 'Q3']:
            partners = self.learner.bandit_choose(get_list=True)

        if self.collab_model == 'lr':
            feats = []
            for ind in self.collab_pop:
                art = self.artifact_cls.individual_to_artifact(ind, self, self.create_kwargs['shape'])
                feats.append(self.get_features(art))
            partners = self.learner.linear_choose_multi(feats)

        return self.addr, partners

    @aiomas.expose
    def get_aesthetic(self):
        return self.aesthetic

    def save_general_info(self):
        sfold = self.artifact_save_folder
        info = {'aesthetic': self.aesthetic,
                'pset_names': self.pset_names,
                'collab_hof_size': self.collab_hof_size,
                'addr': self.addr,
                'collab_model': self.collab_model,
                'collab_iters': self.collab_iters}

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

    def append_coa(self, fb, caest, artifact=None):
        self.collab_arts['caddr'].append(self.caddr)
        self.collab_arts['cinit'].append(self.cinit)
        self.collab_arts['caest'].append(caest)
        self.collab_arts['age'].append(self.age)
        self.collab_arts['fb'].append(fb)
        if fb:
            # Own stuff
            fr = artifact.framings[self.name]
            self.collab_arts['aid'].append(artifact.aid)
            self.collab_arts['val'].append(fr['value'])
            self.collab_arts['nov'].append(fr['novelty'])
            self.collab_arts['eval'].append(artifact.evals[self.name])
            self.collab_arts['mval'].append(fr['max_value'])
            self.collab_arts['nval'].append(fr['norm_value'])
            self.collab_arts['neval'].append(fr['norm_evaluation'])
            # Collaborators stuff
            cfr = artifact.framings[self.caddr]
            self.collab_arts['cval'].append(cfr['value'])
            self.collab_arts['cnov'].append(cfr['novelty'])
            self.collab_arts['ceval'].append(artifact.evals[self.caddr])
            self.collab_arts['cneval'].append(cfr['norm_evaluation'])
            self.collab_arts['cmval'].append(cfr['max_value'])
            self.collab_arts['cnval'].append(cfr['norm_value'])

    def add_collab_evals(self, aid, evals):
        self.collab_evals[aid] = {}
        for addr, e in evals:
            self.own_evals[aid][addr] = e

    def hof2arts(self, hof):
        arts = []
        for ft in hof:
            artifact = GIA(self, ft.image, list(ft), str(ft))
            #artifact.add_eval(self, ft.fitness.values)
            _ = self.evaluate(artifact)
            arts.append((artifact, ft.fitness.values[0]))
        return arts

    def pop2arts(self, pop):
        """Convert a population to a list of artifacts.

        Fitnesses are not saved!
        """
        arts = []
        for ft in pop:
            artifact = GIA(self, ft.image, list(ft), str(ft))
            arts.append(artifact)
        return arts

    def arts2pop(self, arts):
        """Convert a list of given artifacts into a deap population.

        All fitnesses are invalid in the returned population.

        :param arts:
        :return:
        """
        population = []
        for a in arts:
            ind = creator.SuperIndividual(a.framings['function_tree'])
            ind.image = a.obj
            population.append(ind)
        return population

    def init_collab(self):
        """Initialize collaboration by creating the initial population
        and passing it down to other agent after evaluating it.
        """
        #print("{} init collab".format(self.addr))
        init_pop_size = int(self.pop_size / 2)
        population = GIA.initial_population(self,
                                            self.toolbox,
                                            self.pset,
                                            init_pop_size,
                                            method='50-50',
                                            mutate_old=True)
        self.evaluate_population(population)

    def evaluate_population(self, population):
        fitnesses = map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        self.collab_hof.update(population)
        self.collab_pop = population

    @aiomas.expose
    def get_collab_pop(self):
        return self.pop2arts(self.collab_pop)

    async def start_collab(self, max_iter):
        """Start collaboration by passing down initialized collaboration
        population to the collaboration partner and waiting for results.
        """
        self._log(logging.DEBUG,
                  "start collab with {} (i={})"
                  .format(self.addr, self.caddr, max_iter))
        r_agent = await self.connect(self.caddr)
        cagent_init_arts = await r_agent.get_collab_pop()
        cagent_init_pop = self.arts2pop(cagent_init_arts)
        self.collab_pop += cagent_init_pop
        population, i = await self.continue_collab(self.collab_pop, 0)
        self.collab_pop = population
        i = 1  # start from 1 because 0th was the initial population.
        while i <= max_iter:
            arts = self.pop2arts(self.collab_pop)
            ret = await r_agent.rcv_collab(arts, i)
            if i == max_iter - 1:
                ret_arts, hof_arts, i = ret
                pop = self.arts2pop(ret_arts)
                return self.finalize_collab(pop, hof_arts)
            else:
                ret_arts, i = ret
                pop = self.arts2pop(ret_arts)
                population, i = await self.continue_collab(pop, i)
                self.collab_pop = population
        self._log(logging.ERROR,
                  "HÄLÄRMRMRM We should not be here (start_collab)", i)

    async def continue_collab(self, population, iter):
        """Continue collaboration by working on the population.
        """
        #print("{} continue collab iter: {}".format(self.addr, iter))
        #print(population)
        GIA.evolve_population(population, 1, self.toolbox, self.pset,
                              self.collab_hof)
        return population, iter + 1

    def finish_collab(self, population):
        """Make last iteration of collaboration.

        Return population, iteration count and best artifacts to the
        collaboration partner
        """
        #print("{} finish collab".format(self.addr))
        GIA.evolve_population(population, 1, self.toolbox, self.pset,
                              self.collab_hof)
        arts = self.hof2arts(self.collab_hof)
        return population, arts

    def finalize_collab(self, pop, hof_arts):
        """Finalize collaboration from the population and best artifacts
        returned from the last iteration of collaboration by the collaboration
        partner.
        """
        self._log(logging.DEBUG,
                  "finalize collab with {}".format(self.addr, self.caddr))
        self.evaluate_population(pop)
        arts1 = self.hof2arts(self.collab_hof)
        best, ranking = choose_best(hof_arts, arts1, epsilon=0.02)
        return best, ranking

    async def collab_first_iter(self, population, iter):
        """First time an agent receives the collaboration population, it can
        inject some artifacts from its memory to it.

        This might allow for a more meaningful crossover in collaboration.
        """
        pop_size = len(population)
        self_arts = self.stmem.get_artifacts(creator=self.name)
        injected = []
        if len(self_arts) > 0:
            mem_size = min(int(pop_size / 4), len(self_arts))
            mem_arts = np.random.choice(self_arts,
                                        size=mem_size,
                                        replace=False)
            for art in mem_arts:
                individual = creator.SuperIndividual(
                    art.framings['function_tree'])
                self.toolbox.mutate(individual, self.pset)
                del individual.fitness.values
                if individual.image is not None:
                    del individual.image
                injected.append(individual)

        GIA.evolve_population(population, 1, self.toolbox, self.pset,
                              self.collab_hof, injected_inds=injected)
        return population, iter + 1

    @aiomas.expose
    async def rcv_collab(self, arts, iter):
        """Receive collaboration from other agent and continue working on it
        if ``iter < self.collab_iters``.
        """
        #print("{} rcv collab".format(self.addr))
        population = self.arts2pop(arts)
        self.collab_pop = population
        if iter == self.collab_iters - 1:
            pop, hof_arts = self.finish_collab(population)
            ret_arts = self.pop2arts(pop)
            return ret_arts, hof_arts, iter + 1
        # elif iter == 1:
        #     pop, iter = await self.collab_first_iter(population, iter)
        #     ret_arts = self.pop2arts(pop)
        #     return ret_arts, iter
        else:
            pop, iter = await self.continue_collab(population, iter)
            ret_arts = self.pop2arts(pop)
            return ret_arts, iter

    @aiomas.expose
    def rcv_collab_artifact(self, artifact, aesthetic):
        """Receive the final artifact made in collaboration with agent in
        ``addr``.

        The agent in ``addr`` has already computed the results of collaboration,
        and as agents are altruistic, this agent accepts it as it is.
        """
        if artifact is None:
            self.append_coa(False, aesthetic, None)
            return None

        self._log(logging.DEBUG, "Received collab artifact.")
        if self.name not in artifact.evals:
            _ = self.evaluate(artifact)

        fr = artifact.framings[self.name]
        passed = bool(fr['pass_value'] and fr['pass_novelty'])
        if passed:
            # self._log(logging.DEBUG, "Collab artifact passed both thresholds.")
            self.learn(artifact)

        self.append_coa(True, aesthetic, artifact)
        return artifact

    @aiomas.expose
    def clear_collab(self):
        # print("{} clearing collab".format(self.addr))
        super().clear_collab()
        self.collab_hof.clear()
        self.collab_pop = None

    @aiomas.expose
    def get_arts_information(self):
        return self.aesthetic, self.own_arts, self.collab_arts

    @aiomas.expose
    async def act_collab(self, *args, **kwargs):
        """Agent's act when it is time to collaborate.

        :param args:
        :param kwargs:
        :return:
        """
        # Only the collaboration initiators actually act. Others react to them.
        if not self.cinit:
            return

        best, ranking = await self.start_collab(self.collab_iters)

        if best is not None:
            # Change creator name so that it does not come up in memory when
            # creating new individual populations
            best._creator = "{} - {}".format(self.addr, self.caddr)


            #print("Chose best image with ranking {}".format(ranking))
            e, fr = self.evaluate(best)

            #print("Evals: {}".format(best.evals))
            #print("Framings: {}".format(best.framings))
            n = None if fr['novelty'] is None else np.around(fr['novelty'], 2)
            v = np.around(fr['value'], 2)
            e = np.around(e, 2)
            mv = np.around(fr['max_value'], 2)

            r_agent = await self.connect(self.caddr)
            caest = await r_agent.get_aesthetic()

            self._log(logging.INFO,
                      'Collaborated with {} ({}). E={} [V:{} ({}) N:{}]'
                      .format(self.caddr, caest, e, v, mv, n))
            #self.add_artifact(best)

            aid = get_aid(self.addr, self.age, self.aesthetic, v, n, self.caddr, caest)
            best.aid = aid
            art = await r_agent.rcv_collab_artifact(best, self.aesthetic)

            passed = bool(fr['pass_value'] and fr['pass_novelty'])
            if passed:
                # self._log(logging.DEBUG, "Collab artifact passed both thresholds.")
                self.learn(art)

            # Save artifact to save folder
            self.save_artifact(art, pset=self.super_pset, aid=aid)
            self.append_coa(True, caest, art)
            self.last_artifact = art
        else:
            r_agent = await self.connect(self.caddr)
            r_aesthetic = await r_agent.get_aesthetic()
            ret = await r_agent.rcv_collab_artifact(None, self.aesthetic)
            self.append_coa(False, r_aesthetic, None)
            self.last_artifact = None
            self._log(logging.INFO,
                      "could not agree on a collaboration result with {}!"
                      .format(self.caddr))

    @aiomas.expose
    async def act_individual(self, *args, **kwargs):
        """Agent's act for individually made artifacts.
        """
        artifacts = self.invent(self.search_width, n_artifacts=1)
        for artifact, _ in artifacts:
            e, fr = self.evaluate(artifact)
            n = None if fr['novelty'] is None else np.around(fr['novelty'], 2)
            v = np.around(fr['value'], 2)
            self._log(logging.INFO,
                      'Created an individual artifact. E={} (V:{}, N:{})'
                      .format(np.around(e, 2), v, n))
            #self.add_artifact(artifact)

            # Artifact is acceptable if it passes value and novelty thresholds.
            # It is questionable if we need these, however.
            passed = fr['pass_value'] and fr['pass_novelty']
            if passed:
                self._log(logging.DEBUG,
                          "Individual artifact passed thresholds")
                self.learn(artifact)

            # Save artifact to save folder
            aid = get_aid(self.addr, self.age, self.aesthetic, v, n)
            artifact.aid = aid
            self.save_artifact(artifact, pset=self.super_pset, aid=aid)
            # Append to own artifacts
            self.append_oa(artifact)
            self.last_artifact = artifact

    @aiomas.expose
    async def act(self, *args, **kwargs):
        self.last_artifact = None
        self.age += 1
        if self.age % 2 == 1:
            return await self.act_individual(*args, **kwargs)
        else:
            return await self.act_collab(*args, **kwargs)

    @aiomas.expose
    def show_artifact(self, artifact):
        """USE LEARNING MODEL HERE.
        """
        # Do not learn artifact again if it has been shown already.
        if self.name in artifact.framings:
            if self.caddr is not None:
                if self.collab_model == 'Q1':
                    self.learner.update_bandit(artifact.evals[self.name], self.caddr)

                if self.collab_model == 'Q0':
                    own_framings = artifact.framings[self.addr]
                    self_passed = own_framings['pass_novelty'] and own_framings['pass_value']
                    caddr_framings = artifact.framings[self.caddr]
                    caddr_passed = caddr_framings['pass_novelty'] and caddr_framings['pass_value']
                    if self_passed and caddr_passed:
                        self.learner.update_bandit(1, self.caddr)
                    else:
                        self.learner.update_bandit(0, self.caddr)

            return artifact.evals[self.name], artifact.framings[self.name]

        if self.collab_model == 'lr':
            creators = artifact.creator.split(' - ')
            feats = self.get_features(artifact)
            for creator in creators:
                eval = artifact.framings[creator]['norm_evaluation']
                self.learner.update_linear_regression(eval, creator, feats)

        e, fr = self.evaluate(artifact)

        if self.collab_model == 'Q2':
            self.learner.update_bandit(e, artifact.creator)

        # Fixed threshold here.
        if fr['novelty'] > 0.4 and fr['norm_value'] > 0.5:
            self._log(logging.INFO,
                      "Learning (nov={}, nval={}): {}"
                      .format(fr['novelty'], fr['norm_value'], artifact.aid))
            self.learn(artifact)
        return e, fr

    @aiomas.expose
    def rcv_evaluations_from_artifact(self, artifact, evaluations):
        """LEARN MODEL HERE IF NEEDED.

        :param str aid: Artifact ID

        :param dict evaluations:
           Keys are addresses and values are evaluation, framing pairs.
        """
        self._log(logging.INFO,
                  "Got evals from {}: {}".format(artifact.aid, evaluations))
        if self.collab_model == 'random':
            return

        addrs = list(evaluations.keys())
        addrs.remove('creator')
        addrs.remove(self.addr)

        if self.collab_model == 'Q3':
            for addr in addrs:
                self.learner.update_bandit(evaluations[addr][1]['norm_evaluation'], addr)
            return

        if self.collab_model == 'lr':
            for addr in addrs:
                self.learner.update_linear_regression(evaluations[addr][1]['norm_evaluation'],
                                                      addr,
                                                      self.get_features(artifact))
            return


    @aiomas.expose
    def get_last_artifact(self):
        return self.last_artifact

    @aiomas.expose
    def save_arts_info(self):
        sfold = self.artifact_save_folder

        with open(os.path.join(sfold, 'own_arts.pkl'), 'wb') as f:
            pickle.dump(self.own_arts, f)

        # with open(os.path.join(sfold, 'own_evals.pkl'), 'wb') as f:
        #     pickle.dump(self.own_evals, f)

        with open(os.path.join(sfold, 'collab_arts.pkl'), 'wb') as f:
            pickle.dump(self.collab_arts, f)

        # with open(os.path.join(sfold, 'collab_evals.pkl'), 'wb') as f:
        #     pickle.dump(self.collab_evals, f)

        return self.aesthetic, self.own_arts, self.own_evals, self.collab_arts, self.collab_evals

    # @aiomas.expose
    # def close(self, folder=None):
    #     aest_counts = {}
    #     for art in self.collab_arts:
    #         other_aest = art['other_aest']
    #         if other_aest not in aest_counts:
    #             aest_counts[other_aest] = 1
    #         else:
    #             aest_counts[other_aest] += 1
    #     self._log(logging.INFO, str(aest_counts))


class CollabSimulation(Simulation):
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


class CollabEnvironment(StatEnvironment):
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
        self.collab_evals = {}
        self.pref_lists = {'rankings': [],
                           'pairings': []}
        self.age = 0

    def generate_rankings(self, addrs):
        """Generate random rankings by which order agents in ``addrs`` can
        choose their collaboration partners.

        :param list addrs: Addresses of the agents
        :return:
            A sorted list of (addr, ranking) pairs.
        """
        ranks = []
        for a in addrs:
            ranks.append((a, random.random()))
        ranks = sorted(ranks, key=operator.itemgetter(1), reverse=True)
        return ranks

    def find_collab_partner(self, prefs, in_collab):
        """Find the first collaboration partner from ``prefs`` which is not
        ``in_collab`` yet.
        """
        for addr in prefs:
            if addr not in in_collab:
                return addr
        return None

    def match_collab_partners(self, age):
        """Match collaboration partners for the given step.

        Should be called as pre-callback before any agent acts in the current
        step.

        .. seealso::
            :class:`CollabSimulation`
        """
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            return await r_agent.get_collab_prefs()

        async def slave_task2(addr, caddr, cinit):
            r_agent = await self.connect(addr, timeout=5)
            return await r_agent.force_collab(caddr, cinit)

        self.age = age
        collab_step = age % 2 == 0

        if not collab_step:
            return

        self._log(logging.DEBUG,
                  "Matching collaboration partners for step {}.".format(age))

        pref_lists = {}
        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        ret = run(tasks)
        pref_lists.update(ret)
        for addr, pl in ret:
            if addr not in self.pref_lists:
                self.pref_lists[addr] = []
            self.pref_lists[addr].append(pl)
        rankings = self.generate_rankings(pref_lists.keys())
        self.pref_lists['rankings'].append(rankings)
        in_collab = []
        matches = []
        for agent, rank in rankings:
            if agent in in_collab:
                continue
            addr = self.find_collab_partner(pref_lists[agent], in_collab)
            if addr is None:
                print("HÄLÄRM!!! {} did not find a collaboration partner!!"
                      .format(agent))
            else:
                if collab_step:
                    self._log(logging.DEBUG,
                            "Choosing {} to collab with {}.".format(agent, addr))
                in_collab.append(agent)
                in_collab.append(addr)
                matches.append((agent, addr))

        self.pref_lists['pairings'].append(matches)
        #pprint.pprint(self.pref_lists)
        if collab_step:
            for a1, a2 in matches:
                run(slave_task2(a1, a2, cinit=True))
                run(slave_task2(a2, a1, cinit=False))
                self._log(logging.INFO,
                        "{} initializes collab with {}".format(a1, a2))

        if len(matches) != int(len(addrs) / 2):
            print("HÄLÄRMRMRMRMRMR length of matches is not the same as the number of pairs.")

    def clear_collabs(self, *args, **kwargs):
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            return await r_agent.clear_collab()

        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        run(tasks)

    def analyse_arts_inform(self, agent, aest, own_arts, collab_arts):
        loa = len(own_arts['eval'])
        meval = 0
        mnovelty = 0
        mvalue = 0
        if loa > 0:
            meval = np.mean(own_arts['eval'])
            mnovelty = np.mean(own_arts['nov'])
            mvalue = np.mean(own_arts['val'])

        mfound = 0
        mceval = 0
        mcnovelty = 0
        mcvalue = 0
        coa = len(collab_arts['fb'])
        cfound = 0
        if coa > 0 and np.sum(collab_arts['fb']) > 0:
            for i in range(len(collab_arts['eval'])):
                if collab_arts['fb'][i]:
                    mfound += 1
                    mceval += collab_arts['eval'][i]
                    mcnovelty += collab_arts['nov'][i]
                    mcvalue += collab_arts['val'][i]

            cfound = int(mfound)
            if mfound > 0:
                mceval /= mfound
                mcnovelty /= mfound
                mcvalue /= mfound
                mfound /= coa

        self._log(logging.INFO, "{} {} (ind/col): arts={}/{} fb={:.3f} "
                  "e={:.3f}/{:.3f} ({:.3f}) n={:.3f}/{:.3f} ({:.3f}) "
                  "v={:.3f}/{:.3f} ({:.3f})".format(
            agent, aest, loa, cfound, mfound, meval, mceval, meval/mceval,
            mnovelty, mcnovelty, mnovelty/mcnovelty, mvalue, mcvalue,
            mvalue/mcvalue))

    def analyse_all(self):
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            aest, oa, ca = await r_agent.get_arts_information()
            return addr, aest, oa, ca

        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        rets = run(tasks)
        for agent, aest, oa, ca in rets:
            self.analyse_arts_inform(agent, aest, oa, ca)

    def post_cbk(self, *args, **kwargs):
        self.collect_evaluations()
        self.clear_collabs()
        self.analyse_all()

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

        eval_dict = self.collab_evals if self.age % 2 == 0 else self.ind_evals

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
        # print(eval_dict)
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

        with open(os.path.join(sfold, 'collab_evals.pkl'), 'wb') as f:
            pickle.dump(self.collab_evals, f)

        with open(os.path.join(sfold, 'pref_lists.pkl'), 'wb') as f:
            pickle.dump(self.pref_lists, f)

        return rets, self.ind_evals, self.collab_evals

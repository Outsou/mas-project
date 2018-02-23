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
from creamas.util import create_tasks, run
from creamas.rules import RuleLeaf
from creamas.mappers import DoubleLinearMapper

from environments import StatEnvironment

from agents import GPImageAgent, agent_name_parse
from artifacts import GeneticImageArtifact as GIA
from experiments.collab.ranking import choose_best
from learners import MultiLearner
from features import ImageEntropyFeature, ImageComplexityFeature, LinearDiffMapper

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
        self.q_bins = kwargs.pop('q_bins', 20)
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
        if self.collab_model in ['altruistic-Q', 'hedonic-Q', 'lr']:
            self.learner = MultiLearner(list(self.connections),
                                        len(self.R),
                                        e=0)
        elif self.collab_model == 'state-Q':
            self.learner = MultiLearner(list(self.connections),
                                        len(self.R),
                                        e=0,
                                        q_bins=self.q_bins)
        else:
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
                            'rank': [],
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

    def _target_to_state(self):
        raise NotImplementedError("Not implemented.")

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

        if self.collab_model in ['Q0', 'simple-Q', 'altruistic-Q', 'hedonic-Q']:
            partners = self.learner.bandit_choose(get_list=True)

        if self.collab_model == 'state-Q':
            partners = self.learner.q_choose(self._target_to_state(), get_list=True)

        if self.collab_model == 'lr':
            feats = []
            for ind in self.collab_pop:
                art = self.artifact_cls.individual_to_artifact(ind, self, self.create_kwargs['shape'])
                feats.append(self.get_features(art))
            partners = self.learner.linear_choose_multi(feats)

        # if self.collab_model == 'gaussian':
        #     partners = self.learner.gaussian_choose(target, get_list=True)

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

    def append_coa(self, fb, caddr, caest, cinit, artifact=None):
        self.collab_arts['caddr'].append(caddr)
        self.collab_arts['cinit'].append(cinit)
        self.collab_arts['caest'].append(caest)
        self.collab_arts['age'].append(self.age)
        self.collab_arts['fb'].append(fb)
        if fb:
            # Own stuff
            fr = artifact.framings[self.name]
            self.collab_arts['aid'].append(artifact.aid)
            self.collab_arts['rank'].append(artifact.rank)
            self.collab_arts['val'].append(fr['value'])
            self.collab_arts['nov'].append(fr['novelty'])
            self.collab_arts['eval'].append(artifact.evals[self.name])
            self.collab_arts['mval'].append(fr['max_value'])
            self.collab_arts['nval'].append(fr['norm_value'])
            self.collab_arts['neval'].append(fr['norm_evaluation'])
            # Collaborators stuff
            cfr = artifact.framings[caddr]
            self.collab_arts['cval'].append(cfr['value'])
            self.collab_arts['cnov'].append(cfr['novelty'])
            self.collab_arts['ceval'].append(artifact.evals[caddr])
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
    async def select_from_pop(self, population):
        """Use selection to population.

        :param population: Population as Individuals or Artifacts
        """
        are_arts = False
        if issubclass(population[0].__class__, Artifact):
            are_arts = True
            population = self.arts2pop(population)
        self.evaluate_population(population)
        offspring = tools.selBest(population, 1)
        offspring += self.toolbox.select(population, len(population) - 1)
        self.collab_pop = offspring
        if are_arts:
            offspring = self.pop2arts(offspring)
        return offspring


    @aiomas.expose
    def get_collab_pop(self):
        """Return collaboration population as a list of :class:`Artifact` instances.
        """
        return self.pop2arts(self.collab_pop)

    async def start_collab(self, max_iter):
        """Start collaboration by passing down initialized collaboration
        population to the collaboration partner and waiting for results.
        """
        self._log(logging.DEBUG,
                  "start collab with {} (i={})"
                  .format(self.caddr, max_iter))
        r_agent = await self.connect(self.caddr)
        cagent_init_arts = await r_agent.get_collab_pop()
        cagent_init_pop = self.arts2pop(cagent_init_arts)
        selected_cagent_pop = await self.select_from_pop(cagent_init_pop)
        collab_arts = self.pop2arts(self.collab_pop)
        selected_arts = await r_agent.select_from_pop(collab_arts)
        selected_pop = self.arts2pop(selected_arts)
        self.collab_pop = selected_pop + selected_cagent_pop
        random.shuffle(self.collab_pop)
        self.collab_pop = list(map(self.toolbox.clone, self.collab_pop))
        population, i = await self.continue_collab(self.collab_pop, 0,
                                                   use_selection=False)
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

    async def continue_collab(self, population, iter,
                              use_selection=True):
        """Continue collaboration by working on the population.
        """
        #print("{} continue collab iter: {}".format(self.addr, iter))
        #print(population)
        GIA.evolve_population(population, 1, self.toolbox, self.pset,
                              self.collab_hof,
                              use_selection_on_first=use_selection)
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
        best, ranking = choose_best(hof_arts, arts1, epsilon=0.00)
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
            if self.collab_model in ['Q0', 'simple-Q']:
                self.learner.update_bandit(0, self.caddr)
            self.append_coa(False, self.caddr, aesthetic, self.cinit, None)
            return None

        self._log(logging.DEBUG, "Received collab artifact from {}.".format(self.caddr))
        caddr = self.caddr
        cinit = self.cinit
        self.clear_collab()
        if self.name not in artifact.evals:
            _ = self.evaluate(artifact)

        fr = artifact.framings[self.name]
        passed = bool(fr['pass_value'] and fr['pass_novelty'])
        if passed:
            # self._log(logging.DEBUG, "Collab artifact passed both thresholds.")
            self.learn(artifact)

        self.append_coa(True, caddr, aesthetic, cinit, artifact)
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
        """Collaboration act.

        This method returns immediately (and no collaboration is done) if
        ``cinit == False``.
        """
        # Only the collaboration initiators actually act. Others react to them.
        if not self.cinit:
            return

        best, ranking = await self.start_collab(self.collab_iters)

        if best is not None:
            # Change creator name so that it does not come up in memory when
            # creating new individual populations
            best._creator = "{} - {}".format(self.addr, self.caddr)
            best.rank = ranking

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
                      '({}) collab with {} ({}) rank={}. E={} [V:{} ({}) N:{}]'
                      .format(self.aesthetic.upper(), self.caddr, caest.upper(),
                              ranking, e, v, mv, n))
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
            self.append_coa(True, self.caddr, caest, self.cinit, art)
            self.last_artifact = art
        else:
            if self.collab_model in ['Q0', 'simple-Q']:
                self.learner.update_bandit(0, self.caddr)

            r_agent = await self.connect(self.caddr)
            r_aesthetic = await r_agent.get_aesthetic()
            ret = await r_agent.rcv_collab_artifact(None, self.aesthetic)
            self.append_coa(False, self.caddr, r_aesthetic, self.cinit, None)
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
    def show_artifact(self, artifact, collab=False):
        """USE LEARNING MODEL HERE.
        """
        # Do not learn artifact again if it has been shown already.
        if self.name in artifact.framings:
            creators = artifact.creator.split(' - ')
            if len(creators) > 1 and self.addr in creators:
                if self.collab_model == 'simple-Q':
                    self.learner.update_bandit(artifact.evals[self.name], self.caddr)
                if self.collab_model == 'Q0':
                        self.learner.update_bandit(1, self.caddr)

            return artifact.evals[self.name], artifact.framings[self.name]

        if self.collab_model == 'lr':
            creators = artifact.creator.split(' - ')
            feats = self.get_features(artifact)
            for creator in creators:
                eval = artifact.framings[creator]['norm_evaluation']
                self.learner.update_linear_regression(eval, creator, feats)

        e, fr = self.evaluate(artifact)

        if len(artifact.creator.split(' - ')) == 1:
            if self.collab_model == 'hedonic-Q':
                self.learner.update_bandit(e, artifact.creator)
            elif self.collab_model == 'state-Q':
                for i in range(self.q_bins):
                    mapper = self.q_state_mappers[i]
                    val = mapper(fr['feat_val'])
                    self.learner.update_q(val, artifact.creator, i)


            # if self.collab_model == 'gaussian':
            #     self.learner.update_gaussian(fr['value'], artifact.creator)


        # Fixed threshold here.
        if fr['novelty'] > 0.4 and fr['norm_value'] > 0.5:
            self._log(logging.DEBUG,
                      "Learning (nov={}, nval={}): {}"
                      .format(fr['novelty'], fr['norm_value'], artifact.aid))
            self.learn(artifact)
        return e, fr

    @aiomas.expose
    def rcv_evaluations_from_artifact(self, artifact, evaluations):
        """Receive evaluations (done by peers) from an own artifact.

        :param artifact: Artifact the evaluations belong to

        :param dict evaluations:
           Keys are addresses and values are evaluation, framing pairs.
        """
        self._log(logging.DEBUG,
                  "Got evals from {}".format(artifact.aid))
        if self.collab_model == 'random':
            return

        addrs = list(evaluations.keys())
        addrs.remove('creator')
        addrs.remove(self.addr)

        for addr in addrs:
            if self.collab_model == 'altruistic-Q' and artifact.creator == self.addr:
                self.learner.update_bandit(evaluations[addr][1]['norm_evaluation'], addr)
            elif self.collab_model == 'lr':
                feats = self.get_features(artifact)
                self.learner.update_linear_regression(evaluations[addr][1]['norm_evaluation'],
                                                      addr,
                                                      feats)

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

    @aiomas.expose
    def check_drifting(self):
        """Implemented for consistency.
        """
        pass


class DriftingGPCollaborationAgent(GPCollaborationAgent):
    """GP collaboration agent which changes its preferences during the
    simulation's execution.

    At random time steps the preferences will drift considerably. The drifting
    is used to simulate a change in the conceptual thinking of the agent. That
    is, a drifting agents is currently changing its artistic style.

    :param float aesthetic_target:
        The initial target value for aesthetic measure. If float, must be
        between ``aesthetic_bounds``. If it equals to ``"random"``, then
        a random value between given ``aesthetic_bounds`` is chosen.
    :param tuple aesthetic_bounds:
        Bounds for aesthetic's target value as a tuple ``(min, max)``. This
        restricts only the agent's own target values movement. The aesthetic
        measure itself may take values outside these bounds.
    :param float novelty_target:
        The initial target value for novelty. If ``None``, maximizes novelty and
        does not change the target during the agent's life time. Default is
        ``None``.
    :param tuple novelty_bounds:
        Bounds for target novelty value. Ignored if ``novelty_target == None``.
    :param float drifting_prob:
        Probability to start drifting on each iteration. Default is 0.05.
    :param int drifting_speed:
        How many iterations the drifting takes to reach its target.
        Default is 1.

    """
    def __init__(self, *args, **kwargs):
        # Current target value and bounds for the aesthetic function. Different
        # aesthetic functions may have different initial targets and bounds.
        self._aesthetic_target = kwargs.pop('aesthetic_target')
        # Target value bounds for the given aesthetic. The aesthetic's objective
        # values should surpass these bounds at least by some epsilon > 0.
        self.aesthetic_bounds = kwargs.pop('aesthetic_bounds')
        if self._aesthetic_target == 'random':
            self._aesthetic_target = random.uniform(*self.aesthetic_bounds)
        # Noise applied to the aesthetic target on each iteration
        self.aesthetic_noise = kwargs.pop('aesthetic_noise', 0.0)
        # The amount of aesthetic target's drift, this is scaled using the
        # aesthetic bound. The target is drifted by sampling from a normal
        # distribution with mean in the current target the scale determined by
        # the bounds and the drift amount.
        self.aesthetic_drift_amount = kwargs.pop('aesthetic_drift_amount', 0.2)
        # Current target and bounds for the novelty measure
        self.novelty_target = kwargs.pop('novelty_target', None)
        self.novelty_bounds = kwargs.pop('novelty_bounds', (0.1, 0.6))
        # Noise applied to the novelty target on each iteration
        self.novelty_noise = kwargs.pop('novelty_noise', 0.0)
        self.novelty_drift_amount = kwargs.pop('novelty_drift_amount', 0.2)
        # Probability to start drifting on each simulation iteration. (If the
        # targets are already drifting, is omitted.)
        self.drifting_prob = kwargs.pop('drifting_prob', 0.05)
        # How many iterations it takes to drift to the new target.
        self.drifting_speed = kwargs.pop('drifting_speed', 1)
        # How the agent's goal is adjusted in collaboration. Only works with
        # learning model state-Q.
        self.target_adjustment = kwargs.pop('target_adjustment', 'static')
        # Adjusted aesthetic target for the (on-going or previous) collaboration.
        self.adjusted_aesthetic_target = None
        self.adjusted_mapper = None
        # Does curiousness drive the agent to change its own aesthetic target.
        # 'static': no curious behavior, using drifting_prob (if any)
        # 'personal': only accumulating curiosity from own artifacts
        # 'social': accumulating curiosity from both own and others artifacts.
        self.curious_behavior = kwargs.pop('curious_behavior', 'static')
        self.curious_threshold = 1 / (self.drifting_prob * 0.5)
        if self.curious_behavior == 'social':
            # We had to use MATH to come up with this multiplication factor!
            self.curious_threshold = 2.1 * self.curious_threshold
        self.curious_use_distance_scale = kwargs.pop('use_distance_scale', False)
        #print("Scale: {} {}".format(self.curious_use_distance_scale, type(self.curious_use_distance_scale)))
        self.accumulated_curiosity = 0.0
        self._number_of_target_changes = 0
        self.last_target_change = 0

        super().__init__(*args, **kwargs)

        # Set aesthetic target to modify **R** with the current bounds.
        self.aesthetic_target = self._aesthetic_target

        # Are the aesthetic and novelty values currently drifting
        self._is_drifting = False
        # Current drifting target for aesthetic measure. Drifting will end to
        # this target value (or very close to it).
        self._drift_aest_target = None
        # Aesthetic values for the next iterations to reach the current drift
        # target
        self._drift_aest_list = None
        # Current drifting target for novelty measure. Drifting will end to
        # this target value (or very close to it).
        self._drift_novelty_target = None
        # Novelty values for the future iterations to reach the current drift
        # target.
        self._drift_novelty_list = None

        # Modify inherited data structures for saving artifacts.
        self.own_arts['tgt'] = []       # Target for aesthetic value
        self.collab_arts['tgt'] = []    # Unmodified target for aesthetic value
        self.collab_arts['mtgt'] = []   # Modified target for aesthetic value (or same as 'tgt')
        self.collab_arts['ctgt'] = []   # Collaborators (modified) target for aesthetic value

        # Create mappers for bin middle points
        self.q_state_mappers = []
        self.bin_size = (self.aesthetic_bounds[1] - self.aesthetic_bounds[0]) / self.q_bins
        self.bin_borders = []
        for i in range(self.q_bins):
            start = self.aesthetic_bounds[0] + i * self.bin_size
            self.bin_borders.append(start)
            end = start + self.bin_size
            target = (start + end) / 2
            mapper, _ = self._create_mapper(target)
            self.q_state_mappers.append(mapper)
        self.bin_borders.append(self.aesthetic_bounds[1])

    def save_general_info(self):
        sfold = self.artifact_save_folder
        info = {'aesthetic': self.aesthetic,
                'aesthetic_bounds': self.aesthetic_bounds,
                'aesthetic_target': self.aesthetic_target,
                'aesthetic_drift_amount': self.aesthetic_drift_amount,
                'drifting_prob': self.drifting_prob,
                'drifting_speed': self.drifting_speed,
                'target_adjustment': self.target_adjustment,
                'curious_behavior': self.curious_behavior,
                'curious_threshold': self.curious_threshold,
                'pset_names': self.pset_names,
                'collab_hof_size': self.collab_hof_size,
                'addr': self.addr,
                'collab_model': self.collab_model,
                'collab_iters': self.collab_iters}

        with open(os.path.join(sfold, 'general_info.pkl'), 'wb') as f:
            pickle.dump(info, f)

    def get_best_state(self, addr):
        """Returns the best aesthetic target for the addr. Requires state-Q as
        the learning model.
        """
        best_state = self.learner.get_best_state(addr)
        return self.q_state_mappers[best_state]._mid

    @aiomas.expose
    def clear_collab(self):
        super().clear_collab()
        self.adjusted_aesthetic_target = None

    @aiomas.expose
    def force_collab(self, addr, init):
        """Force agent to collaborate with an agent in given address.

        Used by :meth:`CollabEnvironment.match_collab_partners`.

        :param addr: Agent to collaborate with
        :param bool init: Is this agent the initiator in the collaboration.
        """
        super().force_collab(addr, init)

        if self.collab_model == 'state-Q':
            if self.target_adjustment == 'selector' and init:
                self.adjusted_aesthetic_target = self.get_best_state(addr)
                self.adjusted_mapper, _ = self._create_mapper(self.adjusted_aesthetic_target)
                self._log(logging.INFO,
                          "Changed its target from {:.3f} to {:.3f} to match collaborator {}"
                          .format(self.aesthetic_target, self.adjusted_aesthetic_target, addr))
            elif self.target_adjustment == 'selected' and not init:
                self.adjusted_aesthetic_target = self.get_best_state(addr)
                self.adjusted_mapper, _ = self._create_mapper(self.adjusted_aesthetic_target)
                self._log(logging.INFO,
                          "Changed its target from {:.3f} to {:.3f} to match collaborator {}"
                          .format(self.aesthetic_target, self.adjusted_aesthetic_target, addr))
            elif self.target_adjustment == 'both':
                pass

    def _create_mapper(self, new_target):
        """Creates a new double linear mapper.
        """
        if self.aesthetic == 'entropy':
            feat = ImageEntropyFeature
        elif self.aesthetic == 'complexity':
            feat = ImageComplexityFeature
        else:
            raise ValueError("Aesthetic '{}' not recognized"
                             .format(self.aesthetic))
        #dlm = DoubleLinearMapper(feat.MIN, new_target, feat.MAX)
        dlm = LinearDiffMapper(feat.MIN, new_target, feat.MAX)
        return dlm, feat

    def _target_to_state(self):
        """Maps current aesthetic target to a state."""
        dists = [abs(x._mid - self.aesthetic_target) for x in self.q_state_mappers]
        return np.argmin(dists)

    def append_oa(self, artifact):
        super().append_oa(artifact)
        self.own_arts['tgt'].append(self.aesthetic_target)

    def append_coa(self, fb, caddr, caest, cinit, artifact=None, ctgt=None):
        super().append_coa(fb, caddr, caest, cinit, artifact)
        self.collab_arts['tgt'].append(self.aesthetic_target)
        if fb:
            if self.adjusted_aesthetic_target is None:
                self.collab_arts['mtgt'].append(None)
            else:
                self.collab_arts['mtgt'].append(self.adjusted_aesthetic_target)
            self.collab_arts['ctgt'].append(ctgt)

    def get_memory_histogram(self, density=False):
        """Return the histogram of artifacts in the agent's memory.

        Bins of the histogram are the same as the bins for the state-Q.
        """
        feat_vals = []
        for a in self.stmem.get_artifacts():
            feat_val = a.get_feature_value(self.aesthetic)
            if feat_val is not None:
                feat_vals.append(feat_val)
        hgram = np.histogram(feat_vals, self.bin_borders)
        self._log(logging.DEBUG, "{:<10} Memory HG:{}".format(self.aesthetic.upper(), hgram[0]))
        return hgram[0]

    def get_new_curious_target(self, width=4, choose_from=3):
        """Chooses new target using curiosity (based on the memory and state-Q values).

        :param int width:
            How many adjacent states (bins) are considered as suitable targets. Actual number
            of considered states is at most width * 2 + 1.

        :param int choose_from:
            How many choosable states are left from suitable targets after filtering the suitable
            target states using the histogram of memory artifacts.

        :returns:
            New target for the agent.
        """
        def get_current_bin():
            for i, b in enumerate(self.bin_borders):
                if self.aesthetic_target < b:
                    return i - 1
            return i - 1

        b = get_current_bin()
        hg = self.get_memory_histogram()
        lo_bin = b - width if b - width > 0 else 0
        hi_bin = b + width + 1 if b + width + 1 < self.q_bins else self.q_bins
        # Suitable states (bin numbers)
        states = list(range(lo_bin, hi_bin))
        # Suitable states' histogram bins
        filtered_hg = hg[lo_bin:hi_bin]
        # Zip and sort the states using histogram values (from lowest to highest)
        states_hg = sorted(list(zip(states, filtered_hg)), key=lambda x: x[1])
        #print(states_hg)
        # Get 'choose_from' best states from the sorted list
        filtered_states = [e[0] for e in states_hg[:choose_from]]
        #print(filtered_states)
        # Get best state (bin number) from the filtered states using Q-values
        new_target_bin = self.learner.get_best_top_n_state(filtered_states, 4)
        #print(new_target_bin, self.bin_borders[new_target_bin], self.bin_borders[new_target_bin + 1])
        # Randomize a target within the bounds of the chosen bin.
        b_lo = self.bin_borders[new_target_bin]
        b_hi = self.bin_borders[new_target_bin + 1]
        new_target = np.random.uniform(b_lo, b_hi)
        self._number_of_target_changes += 1
        self._log(logging.DEBUG, "{:<10} new curious target: {:.3f} -> {:.3f} (Changes: {})"
                  .format(self.aesthetic.upper(), self.aesthetic_target, new_target,
                          self._number_of_target_changes))
        return new_target

    def get_new_curious_target2(self, remove_n_largest=4, scale_distance=False):
        """Chooses new target using curiosity (based on the memory and state-Q values).

        :param int remove_n_largest:
            Remove *n* bins with the most artifacts in the memory from the new target consideration.

        :param int scale_distance:
            Scale each bins potential for new target with the distance to the current target.

        :returns:
            New target for the agent.
        """
        def get_current_bin():
            for i, b in enumerate(self.bin_borders):
                if self.aesthetic_target < b:
                    return i - 1
            return i - 1

        def get_scale_values(current_bin):
            scale_factor = 1.0 / self.q_bins
            return [1 - (abs(current_bin - i) * scale_factor) for i in range(self.q_bins)]

        current_bin = get_current_bin()
        hg = self.get_memory_histogram()
        states = range(0, self.q_bins)
        # Zip and sort the states using histogram values (from lowest to highest)
        states_hg = sorted(list(zip(states, hg)), key=lambda x: x[1])
        # Remove *n* largest states
        filtered_states_hg = states_hg[:-remove_n_largest]
        free, occupied = self.learner.filter_states()
        # Collect states that are not occupied
        suitable_states = [shg[0] for shg in filtered_states_hg if shg[0] in free]
        # Compute each state's collaboration potential
        states_collab_potential = self.learner.get_best_top_n_state(suitable_states, 4)
        #print(states_collab_potential)
        if len(states_collab_potential) == 0:
            new_target_bin = states_hg[0][0]
        else:
            #sorted_states_collab_potential = sorted(states_collab_potential, key=lambda x: x[1], reverse=True)
            #print(sorted_states_collab_potential)
            if scale_distance:
                scale_factors = get_scale_values(current_bin)
                #self._log(logging.DEBUG, "{}".format(states_collab_potential))
                scaled_collab_potentials = [(i, scale_factors[i] * e) for i, e in states_collab_potential]
                sorted_pots = sorted(scaled_collab_potentials, key=lambda x: x[1], reverse=True)
                #self._log(logging.DEBUG, "{}".format(sorted_pots))
                new_target_bin = sorted_pots[0][0]
            else:
                new_target_bin = states_collab_potential[0][0]

        #print(new_target_bin, self.bin_borders[new_target_bin], self.bin_borders[new_target_bin + 1])
        # Randomize a target within the bounds of the chosen bin.
        b_lo = self.bin_borders[new_target_bin]
        b_hi = self.bin_borders[new_target_bin + 1]
        new_target = np.random.uniform(b_lo, b_hi)
        self._number_of_target_changes += 1
        self._log(logging.DEBUG, "{:<10} new curious target: {}:{:.3f} -> {}:{:.3f} (Changes: {})"
                  .format(self.aesthetic.upper(), current_bin, self.aesthetic_target,
                          new_target_bin, new_target, self._number_of_target_changes))
        return new_target



    @aiomas.expose
    def get_adjusted_aesthetic_target(self):
        return self.adjusted_aesthetic_target

    @aiomas.expose
    def get_aesthetic_target(self):
        return self.aesthetic_target

    @property
    def aesthetic_target(self):
        return self._aesthetic_target

    @aesthetic_target.setter
    def aesthetic_target(self, new_target):
        """Set the aesthetic target of the agent, bounded by
        ``aesthetic_bounds``.

        Aesthetic target is set by removing the first rule from **R** and
        creating a new :class:`RuleLeaf` which added to **R** with weight 1.0.
        """
        if new_target < self.aesthetic_bounds[0]:
            new_target = self.aesthetic_bounds[0]
        if new_target > self.aesthetic_bounds[1]:
            new_target = self.aesthetic_bounds[1]

        self._aesthetic_target = new_target
        dlm, feat = self._create_mapper(new_target)
        self.remove_rule(self.R[0])
        self.add_rule(RuleLeaf(feat(), dlm), 1.0)

    def change_targets(self):
        """Change aesthetic and novelty targets if they are not currently
        drifting.

        You can also directly change the aesthetic target by, e.g.
        ``aesthetic_target = 1.3``. However, ``drift_towards_target`` is
        designed to be used in conjunction with this method. Using this method
        and setting ``aesthetic_target`` during the same run may cause
        unexpected behavior.
        """
        def _get_new_target(cur_target, bounds, drift_amount):
            # Create new drifting target within the bounds
            while True:
                # Scale drifting with absolute bound width.
                bdiff = bounds[1] - bounds[0]
                scale = drift_amount * bdiff
                ddiff = np.random.normal(0.0, scale=scale)
                nt = cur_target + ddiff
                if nt < bounds[0]:
                    return bounds[0]
                if nt > bounds[1]:
                    return bounds[1]
                return nt

        def _compute_drift_waypoints(cur_target, drift_target, n_waypoints):
            """Compute linear drifting waypoints.
            """
            diff = drift_target - cur_target
            wps = []
            for i in range(1, n_waypoints + 1):
                wps.append(cur_target + (i * (diff / n_waypoints)))
            return wps

        # Only create new targets if the targets are not currently drifting.
        if not self._is_drifting:
            self._drift_aest_target = _get_new_target(self.aesthetic_target,
                                                      self.aesthetic_bounds,
                                                      self.aesthetic_drift_amount)
            self._drift_aest_list = _compute_drift_waypoints(self.aesthetic_target,
                                                             self._drift_aest_target,
                                                             self.drifting_speed)
            self._number_of_target_changes += 1
            self._log(logging.DEBUG,
                      "Set AES drifting target to {:.3f} -> {:.3f} ({})".format(
                          self.aesthetic_target,  self._drift_aest_target,
                          self._number_of_target_changes))

            if self.novelty_target is not None:
                self._drift_novelty_target = _get_new_target(self.novelty_target,
                                                             self.novelty_bounds,
                                                             self.novelty_drift_amount)
                self._drift_novelty_list = _compute_drift_waypoints(
                    self.novelty_target,
                    self._drift_novelty_target,
                    self.drifting_speed)
                self._log(logging.DEBUG,
                          "Set NOV drifting target to {:.3f} (from {:.3f})".format(
                              self._drift_aest_target, self.novelty_target))

            self._is_drifting = True

    def drift_towards_targets(self):
        """Drift aesthetic and novelty values towards their current targets.
        """
        if self._drift_aest_list is not None:
            nx_target = self._drift_aest_list.pop(0)
            if self.aesthetic_noise > 0.0:
                nx_target += np.random.normal(0.0, scale=self.aesthetic_noise)
            self.aesthetic_target = nx_target
            self._log(logging.DEBUG, "AES to {:.3f} (target={:.3f}, i={})"
                      .format(self.aesthetic_target,
                              self._drift_aest_target,
                              len(self._drift_aest_list)))
        if self._drift_novelty_list is not None and self.novelty_target is not None:
            self.novelty_target = self._drift_novelty_list.pop(0)
            if self.novelty_noise > 0.0:
                self.novelty_target += np.random.normal(0.0,
                                                        scale=self.novelty_noise)
            self._log(logging.DEBUG, "NOV to {:.3f} (target={:.3f}, i={})"
                      .format(self.novelty_target,
                              self._drift_novelty_target,
                              len(self._drift_novelty_list)))

        if len(self._drift_aest_list) == 0:
            self._is_drifting = False
            self._drift_aest_list = None
            self._drift_novelty_list = None
            #self._log(logging.DEBUG, "Stopped drifting.")

    @aiomas.expose
    def evaluate(self, artifact, use_png_compression=True):
        """Evaluates an artifact based on value (and novelty).

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
                      'feat_val': 0.0,
                      'aest_target': self.aesthetic_target
                      }
                artifact.add_eval(self, evaluation, fr)
                return evaluation, fr

        # Compute the feature value only if it has not been stored in the artifact already.
        try:
            _feat_val = artifact.get_feature_value(self.aesthetic)
            #print(artifact._feat_vals, self.aesthetic)
            if _feat_val is not None:
                #self._log(logging.DEBUG, "Using stored feature value {}".format(_feat_val))
                feat_val = _feat_val
            else:
                feat_val = self.R[0].feat.extract(artifact)
                artifact.add_feature_value(self.aesthetic, feat_val)
                #self._log(logging.DEBUG, "Computed the feature value {}".format(feat_val))
        except Exception:
            print(traceback.format_exc())

        # If we are currently in collaboration and are adjusting our target for the collaboration,
        # then we use the adjusted mapper.
        if self.in_collab and (self.adjusted_aesthetic_target is not None):
            #self._log(logging.DEBUG, "Using adjusted mapper T={}".format(self.adjusted_aesthetic_target))
            value = self.adjusted_mapper(feat_val)
        else:
            value = self.R[0].mapper(feat_val)

        evaluation = value
        if self.novelty_weight != -1:
            novelty = float(self.novelty(artifact))
            evaluation = (1.0 - self.novelty_weight) * value + self.novelty_weight * novelty

        if self.max_value < value:
            self.max_value = value

        norm_value = value / self.max_value
        normalized_evaluation = norm_value
        if self.novelty_weight != -1:
            normalized_evaluation = (1.0 - self.novelty_weight) * norm_value + self.novelty_weight * novelty

        fr = {'value': value,
              'novelty': novelty,
              'pass_novelty': bool(novelty >= self._novelty_threshold) if novelty is not None else True,
              'pass_value': bool(value >= self._value_threshold),
              'max_value': self.max_value,
              'norm_value': norm_value,
              'norm_evaluation': normalized_evaluation,
              'aesthetic': self.aesthetic,
              'feat_val': feat_val,
              'aest_target': self.aesthetic_target
              }
        artifact.add_eval(self, evaluation, fr)
        return evaluation, fr

    @aiomas.expose
    def check_drifting(self):
        """Check if agent should drift in its aesthetic target.
        """
        r = random.random()
        if self.curious_behavior == 'static':
            if r < self.drifting_prob:
                self.change_targets()
                self.last_target_change = self.age
        elif self.accumulated_curiosity > self.curious_threshold:
            self.aesthetic_target = self.get_new_curious_target2(scale_distance=self.curious_use_distance_scale)
            # Curiosity is erased when the target is changed.
            self.accumulated_curiosity = 0.0
            self.last_target_change = self.age
        if self._is_drifting:
            self.drift_towards_targets()

    def accumulate_curiosity(self, feat_val, own_artifact=False):
        """Accumulate curiosity based on observed feature value.
        """
        if own_artifact:
            ldm, _ = self._create_mapper(self.aesthetic_target)
            value = ldm(feat_val)
            n = self.age - self.last_target_change
            curiosity = 1.0 / (value ** n)
            self.accumulated_curiosity += curiosity
            self._log(logging.DEBUG, "Accumulated curiosity {:.3f} now: {:.3f}/{:.3f} ({})"
                      .format(curiosity, self.accumulated_curiosity, self.curious_threshold,
                              self._number_of_target_changes))
        else:
            abs_diff = abs(self.aesthetic_target - feat_val)
            if abs_diff < self.bin_size * 2:
                curiosity = 1 - (abs_diff / (self.bin_size * 2))
                self.accumulated_curiosity += curiosity
                self._log(logging.DEBUG, "Accumulated curiosity {:.3f} now: {:.3f}/{:.3f} ({})"
                          .format(curiosity, self.accumulated_curiosity, self.curious_threshold,
                                  self._number_of_target_changes))

    @aiomas.expose
    def show_artifact(self, artifact, collab=False):
        """USE LEARNING MODEL HERE.
        """
        # Do not learn artifact again if it has been shown already.
        if self.name in artifact.framings:
            creators = artifact.creator.split(' - ')
            if len(creators) > 1 and self.addr in creators:
                if self.collab_model == 'simple-Q':
                    self.learner.update_bandit(artifact.evals[self.name], self.caddr)
                if self.collab_model == 'Q0':
                        self.learner.update_bandit(1, self.caddr)

            return artifact.evals[self.name], artifact.framings[self.name]

        if self.collab_model == 'lr':
            creators = artifact.creator.split(' - ')
            feats = self.get_features(artifact)
            for creator in creators:
                eval = artifact.framings[creator]['norm_evaluation']
                self.learner.update_linear_regression(eval, creator, feats)

        e, fr = self.evaluate(artifact)

        # Accumulate curiosity threshold if applicable
        if self.curious_behavior == 'social':
            self.accumulate_curiosity(fr['feat_val'])

        if len(artifact.creator.split(' - ')) == 1:
            if self.collab_model == 'hedonic-Q':
                self.learner.update_bandit(e, artifact.creator)
            elif self.collab_model == 'state-Q':
                for i in range(self.q_bins):
                    mapper = self.q_state_mappers[i]
                    val = mapper(fr['feat_val'])
                    self.learner.update_q(val, artifact.creator, i)

            # if self.collab_model == 'gaussian':
            #     self.learner.update_gaussian(fr['value'], artifact.creator)

        # Fixed threshold here.
        if fr['novelty'] > 0.4 and fr['norm_value'] > 0.5:
            self._log(logging.DEBUG,
                      "Learning (nov={}, nval={}): {}"
                      .format(fr['novelty'], fr['norm_value'], artifact.aid))
            self.learn(artifact)
        return e, fr

    @aiomas.expose
    async def act_collab(self, *args, **kwargs):
        """Collaboration act.

        This method returns immediately (and no collaboration is done) if
        ``cinit == False``.
        """
        # Only the collaboration initiators actually act. Others react to them.
        if not self.cinit:
            return

        best, ranking = await self.start_collab(self.collab_iters)

        if best is not None:
            # Change creator name so that it does not come up in memory when
            # creating new individual populations
            best._creator = "{} - {}".format(self.addr, self.caddr)
            best.rank = ranking

            #print("Chose best image with ranking {}".format(ranking))
            caddr = self.caddr
            self.clear_collab()
            e, fr = self.evaluate(best)

            if self.curious_behavior in ['personal', 'social']:
                self.accumulate_curiosity(fr['feat_val'], own_artifact=True)

            #print("Evals: {}".format(best.evals))
            #print("Framings: {}".format(best.framings))
            n = None if fr['novelty'] is None else np.around(fr['novelty'], 2)
            v = np.around(fr['value'], 2)
            e = np.around(e, 2)
            mv = np.around(fr['max_value'], 2)

            r_agent = await self.connect(caddr)
            caest = await r_agent.get_aesthetic()

            self._log(logging.INFO,
                      '({}) collab with {} ({}) rank={}. E={} [V:{} ({}) N:{}]'
                      .format(self.aesthetic.upper(), caddr, caest.upper(),
                              ranking, e, v, mv, n))
            #self.add_artifact(best)

            aid = get_aid(self.addr, self.age, self.aesthetic, v, n, caddr, caest)
            best.aid = aid
            art = await r_agent.rcv_collab_artifact(best, self.aesthetic)

            passed = bool(fr['pass_value'] and fr['pass_novelty'])
            if passed:
                # self._log(logging.DEBUG, "Collab artifact passed both thresholds.")
                self.learn(art)

            # Save artifact to save folder
            self.save_artifact(art, pset=self.super_pset, aid=aid)
            self.append_coa(True, caddr, caest, True, art)
            self.last_artifact = art
        else:
            if self.collab_model in ['Q0', 'simple-Q']:
                self.learner.update_bandit(0, self.caddr)

            r_agent = await self.connect(self.caddr)
            r_aesthetic = await r_agent.get_aesthetic()
            ret = await r_agent.rcv_collab_artifact(None, self.aesthetic)
            self.append_coa(False, self.caddr, r_aesthetic, self.cinit, None)
            self.last_artifact = None
            self._log(logging.INFO,
                      "({}) could not agree on a collaboration result with {} ({})!"
                      .format(self.aesthetic.upper(), self.caddr, r_aesthetic.upper))
            self.clear_collab()

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

            if self.curious_behavior in ['personal', 'social']:
                self.accumulate_curiosity(fr['feat_val'], own_artifact=True)
            #self.add_artifact(artifact)

            # Artifact is acceptable if it passes value and novelty thresholds.
            # It is questionable if we need these, however.
            passed = fr['pass_value'] and fr['pass_novelty']
            if passed:
                #self._log(logging.DEBUG,
                #          "Individual artifact passed thresholds")
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
        ret = await super().act(*args, **kwargs)
        hgram = self.get_memory_histogram()
        return ret


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
                self._log(logging.DEBUG, "Choosing {} to collab with {}.".format(agent, addr))
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

        meval_ratio = meval/mceval if mceval > 0.0 else 0.0
        mnovelty_ratio = mnovelty/mcnovelty if mcnovelty > 0.0 else 0.0
        mvalue_ratio = mvalue/mcvalue if mcvalue > 0.0 else 0.0
        self._log(logging.INFO, "{} {:<10} (ind/col): arts={}/{} fb={:.3f} "
                  "e={:.3f}/{:.3f} ({:.3f}) n={:.3f}/{:.3f} ({:.3f}) "
                  "v={:.3f}/{:.3f} ({:.3f})".format(
            agent, aest.upper(), loa, cfound, mfound, meval, mceval, meval_ratio,
            mnovelty, mcnovelty, mnovelty_ratio, mvalue, mcvalue,
            mvalue_ratio))

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
        # Check drifting only after the collaboration step (for the next single artifact steps).
        if self.age % 2 == 0:
            self.check_drifts()
        self.analyse_all()

    def check_drifts(self):
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            await r_agent.check_drifting()

        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs)
        run(tasks)

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

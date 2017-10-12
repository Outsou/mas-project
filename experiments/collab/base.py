"""
Implementations for collaboration subclasses.
"""
import logging
import time
import random
import operator

import numpy as np
from deap import tools, gp, creator, base
import aiomas
from creamas import Simulation
from creamas.util import create_tasks, run

from environments import StatEnvironment

from agents import GPImageAgent
from artifacts import GeneticImageArtifact as GIA
from experiments.collab.ranking import choose_best

__all__ = ['CollaborationBaseAgent',
           'GPCollaborationAgent',
           'CollabEnvironment',
           'CollabSimulation']


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
        partners = list(self.connections.keys())
        if self.collab_model == 'random':
            random.shuffle(partners)
        # TODO: ADD Q-LEARNING ETC. MODELS HERE
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
            print("{}: {} {} {}".format(self.addr, self.in_collab, self.caddr, self.cinit))


class GPCollaborationAgent(CollaborationBaseAgent):
    """Collaboration agent for GP images.
    """
    def __init__(self, *args, **kwargs):
        self.aesthetic = kwargs.pop('aesthetic', '')
        super().__init__(*args, **kwargs)
        self.pop_size = self.create_kwargs['pop_size']
        self.collab_hof_size = 20
        self.collab_hof = tools.HallOfFame(self.collab_hof_size)
        self.collab_pop = None
        self.age = 0

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

    @aiomas.expose
    def get_aesthetic(self):
        return self.aesthetic

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
        print("{} init collab".format(self.addr))
        population = GIA.initial_population(self,
                                            self.toolbox,
                                            self.pset,
                                            self.pop_size,
                                            method='50-50',
                                            mutate_old=True)
        self.evaluate_population(population)

    def evaluate_population(self, population):
        fitnesses = map(self.toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        self.collab_hof.update(population)
        self.collab_pop = population

    async def start_collab(self, max_iter):
        """Start collaboration by passing down initialized collaboration
        population to the collaboration partner and waiting for results.
        """
        print("{} start collab with {} iters".format(self.addr, max_iter))
        r_agent = await self.connect(self.caddr)
        iter = 1
        while iter <= max_iter:
            arts = self.pop2arts(self.collab_pop)
            ret = await r_agent.rcv_collab(arts, iter)
            if iter == max_iter - 1:
                ret_arts, hof_arts, iter = ret
                pop = self.arts2pop(ret_arts)
                return self.finalize_collab(pop, hof_arts)
            else:
                ret_arts, iter = ret
                pop = self.arts2pop(ret_arts)
                population, iter = await self.continue_collab(pop, iter)
                self.collab_pop = population
        #print("plookplpl", iter)

    async def continue_collab(self, population, iter):
        """Continue collaboration by working on the population.
        """
        print("{} continue collab iter: {}".format(self.addr, iter))
        #print(population)
        GIA.evolve_population(population, 1, self.toolbox, self.pset,
                              self.collab_hof)
        return population, iter + 1

    def finish_collab(self, population):
        """Make last iteration of collaboration.

        Return population, iteration count and best artifacts to the
        collaboration partner
        """
        print("{} finish collab".format(self.addr))
        GIA.evolve_population(population, 1, self.toolbox, self.pset,
                              self.collab_hof)
        arts = self.hof2arts(self.collab_hof)
        return population, arts

    def finalize_collab(self, pop, hof_arts):
        """Finalize collaboration from the population and best artifacts
        returned from the last iteration of collaboration by the collaboration
        partner.
        """
        print("{} finalize collab".format(self.addr))
        self.evaluate_population(pop)
        arts1 = self.hof2arts(self.collab_hof)
        best, ranking = choose_best(hof_arts, arts1, epsilon=0.02)
        return best, ranking

    @aiomas.expose
    async def rcv_collab(self, arts, iter):
        """Receive collaboration from other agent and continue working on it
        if ``iter < self.collab_iters``.
        """
        print("{} rcv collab".format(self.addr))
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
    def rcv_collab_artifact(self, addr, artifact):
        """Receive the final artifact made in collaboration with agent in
        ``addr``.

        The agent in ``addr`` has already computed the results of collaboration,
        and as agents are altruistic, this agent accepts it as it is.
        """
        pass

    @aiomas.expose
    def clear_collab(self):
        print("{} clearing collab".format(self.addr))
        super().clear_collab()
        self.collab_hof.clear()
        self.collab_pop = None

    @aiomas.expose
    async def collab_act(self, *args, **kwargs):
        """Agent's act when it is time to collaborate.

        :param args:
        :param kwargs:
        :return:
        """
        # Only the collaboration initiators actually act. Others react to them.
        if not self.cinit:
            return

        self.init_collab()
        best, ranking = await self.start_collab(self.collab_iters)

        if best is not None:
            print("Chose best image with ranking {}".format(ranking))
            e, fr = self.evaluate(best)
            print("Evals: {}".format(best.evals))
            n = None if fr['novelty'] is None else np.around(fr['novelty'],
                                                             2)
            self._log(logging.INFO,
                      'Collaborated with {}. E={} (V:{}, N:{})'
                      .format(self.caddr, np.around(e, 2),
                              np.around(fr['value'], 2), n))
            self.add_artifact(best)

            if e >= self._own_threshold:
                self.learn(best)

            # Save artifact to save folder
            r_agent = await self.connect(self.caddr)
            r_aesthetic = await r_agent.get_aesthetic()
            aid = "{:0>5}_{}-{}".format(self.age, self.aesthetic, r_aesthetic)
            self.save_artifact(best, pset=self.super_pset, aid=aid)
        else:
            self._log(logging.INFO,
                      "could not agree on a collaboration result with {}!"
                      .format(self.caddr))

    @aiomas.expose
    async def individual_act(self, *args, **kwargs):
        """Agent's act for individually made artifacts.
        """
        artifacts = self.invent(self.search_width, n_artifacts=1)
        for artifact, _ in artifacts:
            e, fr = self.evaluate(artifact)
            n = None if fr['novelty'] is None else np.around(fr['novelty'], 2)
            self._log(logging.INFO,
                      'Created an individual artifact. E={} (V:{}, N:{})'
                      .format(np.around(e, 2), np.around(fr['value'], 2), n))
            self.add_artifact(artifact)

            if e >= self._own_threshold:
                self.learn(artifact)

            # Save artifact to save folder
            self.save_artifact(artifact)

    @aiomas.expose
    async def act(self, *args, **kwargs):
        self.age += 1
        if self.age % 2 == 1:
            return await self.individual_act(*args, **kwargs)
        else:
            return await self.collab_act(*args, **kwargs)




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
        super().__init__(*args, **kwargs)

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

        self._log(logging.DEBUG,
                  "Matching collaboration partners for step {}.".format(age))
        pref_lists = {}
        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        ret = run(tasks)
        pref_lists.update(ret)
        rankings = self.generate_rankings(pref_lists.keys())
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
                self._log(logging.DEBUG,
                          "Choosing {} to collab with {}.".format(agent, addr))
                in_collab.append(agent)
                in_collab.append(addr)
                matches.append((agent, addr))

        for a1, a2 in matches:
            run(slave_task2(a1, a2, cinit=True))
            run(slave_task2(a2, a1, cinit=False))
            print("{} initializes collab with {}".format(a1, a2))

        if len(matches) != int(len(addrs) / 2):
            print("HÄLÄRMRMRMRMRMR length of matches is not the same as the number of pairs.")

    def clear_collabs(self, *args, **kwargs):
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            return await r_agent.clear_collab()

        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        run(tasks)


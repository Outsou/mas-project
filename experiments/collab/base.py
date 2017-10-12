"""
Implementations for collaboration subclasses.
"""
import logging
import time
import random
import operator
import os

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
        print("artifacts in memory: {} ({})".format(len(self.get_artifacts()),
                                                    len(self.artifacts)))

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

        # Data structures to keep information about the created artifacts.
        # Mainly evaluations, e.g. value and novelty for own artifacts,
        # but also for collaborated artifacts the partner's evaluations and
        # their aesthetic functions.
        self.own_arts = []
        self.collab_arts = []

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
        #print("{} init collab".format(self.addr))
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
        self._log(logging.DEBUG,
                  "start collab with {} (i={})"
                  .format(self.addr, self.caddr, max_iter))
        r_agent = await self.connect(self.caddr)
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
    def rcv_collab_artifact(self, artifact, addr, aesthetic, collab_passed):
        """Receive the final artifact made in collaboration with agent in
        ``addr``.

        The agent in ``addr`` has already computed the results of collaboration,
        and as agents are altruistic, this agent accepts it as it is.
        """
        if artifact is None:
            self.collab_arts.append({'addr': self.caddr,
                                     'init': self.cinit,
                                     'found_best': False,
                                     'other_aest': aesthetic,
                                     'self_pass': False,
                                     'other_pass': False,
                                     'value': None,
                                     'novelty': None,
                                     'evaluation': None,
                                     'i': self.age})
            return False

        self._log(logging.DEBUG, "Received collab artifact.")
        if self.name not in artifact.evals:
            _ = self.evaluate(artifact)

        fr = artifact.framings[self.name]
        passed = bool(fr['pass_value'] and fr['pass_novelty'])
        if passed and collab_passed:
            self._log(logging.DEBUG, "Collab artifact passed both thresholds.")
            self.learn(artifact)

        self.collab_arts.append({'addr': self.caddr,
                                 'init': self.cinit,
                                 'found_best': True,
                                 'other_aest': aesthetic,
                                 'self_pass': passed,
                                 'other_pass': collab_passed,
                                 'value': fr['value'],
                                 'novelty': fr['novelty'],
                                 'evaluation': artifact.evals[self.name],
                                 'i': self.age})

        return passed

    @aiomas.expose
    def clear_collab(self):
        print("{} clearing collab".format(self.addr))
        super().clear_collab()
        self.collab_hof.clear()
        self.collab_pop = None

    @aiomas.expose
    def get_arts_information(self):
        return self.own_arts, self.collab_arts

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

        self.init_collab()
        best, ranking = await self.start_collab(self.collab_iters)

        if best is not None:
            # Change creator name so that it does not come up in memory when
            # creating new individual populations
            best._creator = "{} - {}".format(self.addr, self.caddr)

            #print("Chose best image with ranking {}".format(ranking))
            e, fr = self.evaluate(best)
            #print("Evals: {}".format(best.evals))
            #print("Framings: {}".format(best.framings))
            n = None if fr['novelty'] is None else np.around(fr['novelty'],
                                                             2)
            self._log(logging.INFO,
                      'Collaborated with {}. E={} (V:{}, N:{})'
                      .format(self.caddr, np.around(e, 2),
                              np.around(fr['value'], 2), n))
            self.add_artifact(best)

            passed = bool(fr['pass_value'] and fr['pass_novelty'])

            r_agent = await self.connect(self.caddr)
            r_aesthetic = await r_agent.get_aesthetic()
            ret = await r_agent.rcv_collab_artifact(best, self.addr, self.aesthetic, passed)

            self.collab_arts.append({'addr': self.caddr,
                                     'init': self.cinit,
                                     'found_best': True,
                                     'other_aest': r_aesthetic,
                                     'self_pass': passed,
                                     'other_pass': ret,
                                     'value': fr['value'],
                                     'novelty': fr['novelty'],
                                     'evaluation': e,
                                     'i': self.age})

            if passed and ret:
                self._log(logging.DEBUG, "Collab artifact passed both thresholds.")
                self.learn(best)

            # Save artifact to save folder
            aid = "{:0>5}_{}-{}".format(self.age, self.aesthetic, r_aesthetic)
            self.save_artifact(best, pset=self.super_pset, aid=aid)
        else:
            r_agent = await self.connect(self.caddr)
            r_aesthetic = await r_agent.get_aesthetic()
            ret = await r_agent.rcv_collab_artifact(None, self.addr, self.aesthetic, False)
            self.collab_arts.append({'addr': self.caddr,
                                     'init': self.cinit,
                                     'found_best': False,
                                     'other_aest': r_aesthetic,
                                     'self_pass': False,
                                     'other_pass': False,
                                     'value': None,
                                     'novelty': None,
                                     'evaluation': None,
                                     'i': self.age})
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
            self.add_artifact(artifact)

            # Artifact is acceptable if it passes value and novelty thresholds.
            # It is questionable if we need these, however.
            passed = fr['pass_value'] and fr['pass_novelty']
            if passed:
                self._log(logging.DEBUG, "Individual artifact passed thresholds")
                self.learn(artifact)

            self.own_arts.append({'self_pass': True,
                                  'value': fr['value'],
                                  'novelty': fr['novelty'],
                                  'evaluation': e,
                                  'i': self.age})

            # Save artifact to save folder
            aid = "{:0>5}_{}_v{}_n{}".format(self.age, self.aesthetic, v, n)
            self.save_artifact(artifact, pset=self.super_pset, aid=aid)

    @aiomas.expose
    async def act(self, *args, **kwargs):
        self.age += 1
        if self.age % 2 == 1:
            return await self.act_individual(*args, **kwargs)
        else:
            return await self.act_collab(*args, **kwargs)

    @aiomas.expose
    def save_arts_info(self):
        oa = self.own_arts
        ca = self.collab_arts
        sfold = self.artifact_save_folder
        writes = []
        # Own artifacts
        if len(oa) > 0:
            es = ('evaluations.txt', [a['evaluation'] for a in oa])
            ns = ('novelties.txt', [a['novelty'] for a in oa])
            vs = ('values.txt', [a['value'] for a in oa])
            ps = ('passes.txt', [a['self_pass'] for a in oa])
            ages = ('iters.txt', [a['i'] for a in oa])
            writes += [es, ns, vs, ps]

            # Running means
            mes = ('mevaluations.txt', [sum(es[1][:i])/(i+1) for i in range(len(oa))])
            mns = ('mnovelties.txt', [sum(ns[1][:i])/(i+1) for i in range(len(oa))])
            mvs = ('mvalues.txt', [sum(vs[1][:i])/(i+1) for i in range(len(oa))])
            mps = ('mpasses.txt', [sum(ps[1][:i])/(i+1) for i in range(len(oa))])
            writes += [mes, mns, mvs, mps]
        else:
            writes += [None, None, None, None]

        # Collaborated artifacts
        if len(ca) > 0:
            ces = ('cevaluations.txt', [a['evaluation'] for a in ca])
            cns = ('cnovelties.txt', [a['novelty'] for a in ca])
            cvs = ('cvalues.txt', [a['value'] for a in ca])
            cps = ('cpasses.txt', [a['self_pass'] for a in ca])
            cages = ('citers.txt', [a['i'] for a in ca])
            writes += [ces, cns, cvs, cps, cages]

            # Running means
            mces = ('mcevaluations.txt', [sum(ces[1][:i])/(i+1) for i in range(len(oa))])
            mcns = ('mcnovelties.txt', [sum(cns[1][:i])/(i+1) for i in range(len(oa))])
            mcvs = ('mcvalues.txt', [sum(cvs[1][:i])/(i+1) for i in range(len(oa))])
            mcps = ('mcpasses.txt', [sum(cps[1][:i])/(i+1) for i in range(len(oa))])
            writes += [mces, mcns, mcvs, mcps]
        else:
            writes += [None, None, None, None]

        for w in writes:
            if w is not None:
                fname, data = w
                fpath = os.path.join(sfold, fname)
                with open(fpath, 'w') as f:
                    for d in data:
                        f.write("{}\n".format(d))

        return writes


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

    def analyse_arts_inform(self, agent, own_arts, collab_arts):
        # print("Analysing {}".format(agent))

        meval = 0.0
        mnovelty = 0.0
        mvalue = 0.0
        mpassed = 0.0
        loa = len(own_arts)
        if loa > 0:
            for a in own_arts:
                mpassed += a['self_pass']
                meval += a['evaluation']
                mnovelty += a['novelty']
                mvalue += a['value']

            mpassed /= loa
            meval /= loa
            mnovelty /= loa
            mvalue /= loa
            print("{} own: p={:.3f} e={:.3f} n={:.3f} v={:.3f}"
                  .format(agent, mpassed, meval, mnovelty, mvalue))

        mceval = 0.0
        mcnovelty = 0.0
        mcvalue = 0.0
        mspassed = 0.0
        mopassed = 0.0
        mbpassed = 0.0
        mfound = 0.0
        coa = len(collab_arts)
        if coa > 0:
            for a in collab_arts:
                if a['found_best']:
                    mfound += 1
                    mspassed += a['self_pass']
                    mopassed += a['other_pass']
                    mbpassed += a['self_pass'] and a['other_pass']
                    mceval += a['evaluation']
                    mcnovelty += a['novelty']
                    mcvalue += a['value']

            if mfound > 0:
                mspassed /= mfound
                mopassed /= mfound
                mbpassed /= mfound
                mceval /= mfound
                mcnovelty /= mfound
                mcvalue /= mfound
                mfound /= coa
                print("{} collab: fb={:.3f}, ps={:.3f} po={:.3f} pb={:.3f} e={:.3f} n={:.3f} v={:.3f}"
                      .format(agent, mfound, mspassed, mopassed, mbpassed, mceval, mcnovelty, mcvalue))
                print("{} ratio (ind/col): arts={:.3f} ({}/{}) e={:.3f} n={:.3f} v={:.3f}"
                      .format(agent, loa/coa, loa, coa, meval/mceval, mnovelty/mcnovelty, mvalue/mcvalue))
            else:
                print("{} collab: fb=0".format(agent))

    def analyse_all(self):
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            oa, ca = await r_agent.get_arts_information()
            return addr, oa, ca

        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        rets = run(tasks)
        for agent, oa, ca in rets:
            self.analyse_arts_inform(agent, oa, ca)

    def post_cbk(self, *args, **kwargs):
        self.clear_collabs()
        self.analyse_all()

    def save_artifact_info(self, path):
        async def slave_task(addr):
            r_agent = await self.connect(addr, timeout=5)
            ret = await r_agent.save_arts_info()
            return addr, ret

        self._log(logging.INFO,
                  "Saving all info.")

        addrs = self.get_agents(addr=True)
        tasks = create_tasks(slave_task, addrs, flatten=False)
        rets = run(tasks)
        collected = {}
        for agent, ret in rets:
            for r in ret:
                if r is not None:
                    if r[0] not in collected:
                        collected[r[0]] = []
                    rs = [e for e in r[1] if e is not None]
                    llen = len(rs)
                    collected[r[0]].append((sum(rs), llen))

        for k, v in collected.items():
            ssum = sum([s[0] for s in v])
            lens = sum([s[1] for s in v])
            mm = ssum / lens
            fpath = os.path.join(path, k)
            with open(fpath, 'w') as f:
                f.write("{}\n".format(mm))
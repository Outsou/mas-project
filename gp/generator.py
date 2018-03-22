"""Generator creating evolutionary art using genetic programming.

The generated artifacts are instances of :class:`GPImageArtifact`.
"""
import deap
import cv2

import numpy as np

from gp.artifact import GPImageArtifact

class GPImageGenerator():
    """A generator class producing instances of :class:`GPImageArtifact` using genetic programming.

    The generator uses `DEAP <https://deap.readthedocs.io/en/master/>`_ in its internal operation.

    Generator class can be used in two different manners:

        * calling only the static functions of the class, or
        * creating a new instance of the class associated with a specific agent and calling
          :meth:`GPImageGenerator.generate`.

    """
    def __init__(self, agent, pset, generations, pop_size, shape=(32, 32), evaluate_func=None,
                 super_pset=None, **kwargs):
        """
        Most of these basic initialization parameters can be overridden in
        :meth:`GPImageGenerator.generate`.

        :param agent:
            An agent this generator is associated to.
        :param pset:
            DEAP's primitive set which the agent uses to create the images.
        :param generations:
            Number of generations to evolve
        :param pop_size:
            Population size.
        :param shape:
            Shape of the produced images during the evolution. The resolution can be (indefinitely)
            up-scaled for the accepted artifacts.
        :param evaluate_func:
            A function used to evaluate each image, if ``None``, ``agent.evaluate`` is used.
            Function should accept one argument, a :class:`GPImageArtifact` and return an evaluation
            of an artifact. Evaluation is supposed to be maximized.
        :param super_pset:
            In a case an agent may create artifacts in conjunction with other agents, ``super_pset``
            should contain all the primitives all the agents may use.
        :param kwargs:
        """
        self.agent = agent
        self.pset = pset
        self.generations = generations
        self.pop_size = pop_size
        self.shape = shape
        if evaluate_func is None:
            self.evaluate_artifact = self.agent.evaluate
        else:
            self.evaluate_artifact = evaluate_func
        if super_pset is None:
            self.super_pset = pset
        else:
            self.super_pset = super_pset

        self.toolbox = self.agent.toolbox

    @staticmethod
    def individual_to_artifact(agent, individual, shape, pset=None, bw=True):
        """Convert DEAP´s ``individual`` to :class:`GPImageArtifact`.

        This will create the inherent image object, if it is not already present in the
        ``individual``. If individual has already an image associated with it, that image is used
        and ``shape`` and ``bw`` parameters are omitted.

        :param agent:
            Creator of the image.
        :param individual:
            Function (DEAP´s individual) of the image.
        :param shape:
            Shape of the returned image.
        :param pset:
            DEAP's primitive set used to compile the individual.
        :param bw:
            If ``True``, ``func`` is assumed to represent RGB image, otherwise it is assumed to
            be a greyscale image.
        :return:
            :class:`GPImageArtifact`
        """
        if individual.image is None:
            try:
                func = deap.gp.compile(individual, pset)
            except MemoryError:
                return None
            func_str = str(individual)
            image = GPImageArtifact.func2image(func, shape, bw=bw)
            individual.image = image

        artifact = GPImageArtifact(agent, individual.image, individual, str(individual))
        return artifact

    def evaluate_individual(self, individual, shape):
        """Evaluates a DEAP individual.

        This method inherently changes the individual to :class:`GPImageArtifact` for
        the evaluation.

        :param individual:
            The individual to be evaluated.
        :param agent:
            The agent whose evaluation function is used.
        :param shape:
            Shape of the image.
        :return:
            The evaluation.
        """
        pset = self.super_pset
        artifact = GPImageGenerator.individual_to_artifact(self.agent, individual, shape, pset)

        if artifact is None:
            return -1,

        return self.evaluate_artifact(artifact)

    @staticmethod
    def png_compression_ratio(artifact):
        """Compute png compression ratio for the image of the given artifact.

        PNG compression ratio: size(png) / size(bmp)

        If ratio is low (< 0.08), then the image can be seen as uninteresting.
        """
        img = artifact.obj
        bmp_size = (img.shape[0] * img.shape[1]) + 1078
        if len(img.shape) == 3:
            bmp_size = (img.shape[0] * img.shape[1] * img.shape[2]) + 54
        _, buf = cv2.imencode('.png', img)
        png_size = len(buf)
        del buf
        ratio = png_size / bmp_size
        return ratio

    @staticmethod
    def evolve_population(population, generations, toolbox, pset, hall_of_fame,
                          cxpb=0.75, mutpb=0.25, injected_inds=[],
                          use_selection_on_first=True):
        """
        Evolves a population of individuals. Applies elitist (k=1) in addition
        to toolbox's selection strategy to the individuals.

        :param population:
            A list containing the individuals of the population.
        :param generations:
            Number of generations to be evolved.
        :param toolbox:
            DEAP toolbox with the necessary functions.
        :param pset:
            DEAP primitive set used during the evolution.
        """
        pop_len = len(population)
        population += injected_inds
        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        hall_of_fame.update(population)

        for g in range(generations):
            if not use_selection_on_first and g == 0:
                offspring = list(map(toolbox.clone, population))
            else:
                # Select the next generation individuals with elitist (k=1) and
                # toolboxes selection method
                offspring = deap.tools.selBest(population, 1)
                offspring += toolbox.select(population, pop_len - 1)
                # Clone the selected individuals
                offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring

            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < cxpb:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    if child1.image is not None:
                        del child1.image
                    if child2.image is not None:
                        del child2.image

            for mutant in offspring:
                if np.random.random() < mutpb:
                    toolbox.mutate(mutant, pset)
                    del mutant.fitness.values
                    if mutant.image is not None:
                        del mutant.image

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = map(toolbox.evaluate, invalid_ind)
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            population[:] = offspring
            # Update hall of fame with new population.
            hall_of_fame.update(population)

    @staticmethod
    def create_population(size, pset, generation_method):
        """Creates a population of randomly generated individuals.

        :param size:
            The size of the generated population.
        :param pset:
            The primitive set used in individual generation.
        :param generation_method:
            Generation method to create individuals, e.g.
            :meth:`deap.gp.genHalfAndHalf`.
        :return:
            A list containing the generated population as DEAP individuals.
        """
        GPImageGenerator.init_creator(pset)
        pop_toolbox = deap.base.Toolbox()
        pop_toolbox.register("expr", generation_method, pset=pset, min_=2, max_=6)
        pop_toolbox.register("individual", deap.tools.initIterate, deap.creator.Individual,
                             pop_toolbox.expr)
        pop_toolbox.register("population", deap.tools.initRepeat, list, pop_toolbox.individual)
        return pop_toolbox.population(size)

    @staticmethod
    def init_creator(pset):
        """Initializes the DEAP :class:`deap.creator` to use the wanted primitive set and maximizing
        fitness.
        """
        deap.creator.create("FitnessMax", deap.base.Fitness, weights=(1.0,))
        deap.creator.create("Individual", deap.gp.PrimitiveTree, fitness=deap.creator.FitnessMax,
                            pset=pset, image=None)


    @staticmethod
    def initial_population(agent, toolbox, pset, pop_size, method='random',
                           mutate_old=True):
        """Create initial population for a new artifact invention/collaboration
        process.

        :param agent:
            Agent which creates the population.
        :param pset:
            pset for the population's new individuals
        :param pop_size:
            Size of the population.
        :param method:
            Either '50-50' or 'random'. If '50-50' takes (at most) half of the
            initial artifacts from the agent's memory and creates others. If
            'random' creates all individuals.
        :param bool mutate_old:
            If ``True``, forces mutation on the artifacts acquired from
            the memory.
        :return: Created population
        """
        # Return at most half old artifacts and at least half new ones.
        if method == '50-50':
            population = []
            self_arts = agent.stmem.get_artifacts(creator=agent.name)
            if len(self_arts) > 0:
                mem_size = min(int(pop_size / 2), len(self_arts))
                mem_arts = np.random.choice(self_arts,
                                            size=mem_size,
                                            replace=False)
                for art in mem_arts:
                    # Super individual has all primitives, so no need to call
                    # agent specific individual creator (with agent specific
                    # pset).
                    individual = deap.creator.SuperIndividual(
                        art.framings['function_tree'])
                    # Force mutate artifacts from the memory
                    if mutate_old:
                        toolbox.mutate(individual, pset)
                        del individual.fitness.values
                        if individual.image is not None:
                            del individual.image
                    population.append(individual)

            if len(population) < pop_size:
                population += GPImageGenerator.create_population(
                    pop_size - len(population), pset, gp.genHalfAndHalf)
            return population
        # Return random population
        return GPImageGenerator.create_population(pop_size, pset, deap.gp.genHalfAndHalf)

    def generate(self, artifacts=1, generations=None, pset=None, pop_size=None, shape=None,
                 init_method='50-50'):
        """
        Generate new artifacts.

        :param int artifacts:
            The number of the best artifacts to be returned.
        :param int generations:
            Number of generations to be evolved. If ``None`` uses initialization parameters.
        :param pset:
            DEAP's primitive set used to create the individuals.
        :param int pop_size:
            DEAP population size
        :param tuple shape:
            Shape of the created images. This heavily affects the execution time.
        :return:
            A list of generated class:`GPImageArtifact` objects. The artifacts returned do not
            have their framing information (evaluation, etc.) filled.
        """
        # Initialize parameters
        generations = generations if generations is not None else self.generations
        pset = pset if pset is not None else self.pset
        pop_size = pop_size if pop_size is not None else self.pop_size
        pset = shape if shape is not None else self.shape

        hall_of_fame = deap.tools.HallOfFame(artifacts)
        # Create initial population
        population = GPImageGenerator.initial_population(self.agent,
                                                         self.toolbox,
                                                         pset,
                                                         pop_size,
                                                         init_method)
        self.toolbox.register("evaluate", GPImageGenerator.evaluate_individual, shape=shape)
        # Evolve population for the number of generations.
        GPImageGenerator.evolve_population(population, generations, self.toolbox, pset,
                                           hall_of_fame)
        arts = []
        for ft in hall_of_fame:
            artifact = GPImageArtifact(self.agent, ft.image, list(ft), str(ft))
            arts.append((artifact, None))
        return arts

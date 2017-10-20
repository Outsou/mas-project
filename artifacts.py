import os
from io import BytesIO

from creamas.core.artifact import Artifact

from deap import tools, creator, base
import deap.gp as gp
from deap import algorithms
import cv2

import logging
import numpy as np
from scipy import misc
import matplotlib.pyplot as plt
from matplotlib import cm

import traceback


class DummyArtifact(Artifact):
    """A dummy artifact used for testing purposes. Generates feature values for
    an imaginary artifact.
    """
    def __init__(self, creator, obj):
        super().__init__(creator, obj, domain='dummy')

    @staticmethod
    def max_distance(create_kwargs):
        """Max distance is the distance between a vector of zeros and ones.
        """
        return np.linalg.norm(np.ones(create_kwargs['length']))

    @staticmethod
    def distance(artifact1, artifact2):
        """Euclidian distance between two vectors.
        """
        obj1 = np.array(artifact1.obj)
        obj2 = np.array(artifact2.obj)
        return np.linalg.norm(obj1 - obj2)

    @staticmethod
    def create(length):
        """Creates a random vector of features.
        """
        return np.random.rand(length)

    @staticmethod
    def invent(n, agent, create_kwargs, n_artifacts=1):
        """Creates n artifacts and returns the best one.
        """
        def add_feature_framings(artifact):
            artifact.framings['features'] = {}
            for i in range(len(artifact.obj)):
                name = 'dummy_' + str(i)
                artifact.framings['features'][name] = artifact.obj[i]

        arts = []
        for _ in range(n):
            obj = DummyArtifact.create(**create_kwargs)
            artifact = DummyArtifact(agent, obj)
            add_feature_framings(artifact)
            eval, _ = agent.evaluate(artifact)
            arts.append(artifact)
        arts = sorted(arts, key=lambda art: art.evals[agent.addr], reverse=True)[:n_artifacts]
        return [(art, None) for art in arts]


class GeneticImageArtifact(Artifact):
    def __init__(self, creator, obj, function_tree, string_repr=None):
        super().__init__(creator, obj, domain='image')
        self.framings['function_tree'] = function_tree
        self.framings['string_repr'] = string_repr
        self.png_compression_done = False
        # Artifact ID #
        self.aid = None
        self.rank = None

    @staticmethod
    def artifact_from_file(fname, pset):
        """Recreate an individual from a string saved into a file.
        """
        s = ""
        with open(fname, 'r') as f:
            s = f.readline()
        s = s.strip()
        individual = gp.PrimitiveTree.from_string(s, pset)
        return individual

    @staticmethod
    def resave_with_resolution(fname, pset, color_map, shape=(1000, 1000)):
        """Resave an individual saved as a string into a file with given
        color mapping and resolution.
        """
        individual = GeneticImageArtifact.artifact_from_file(fname, pset)
        func = gp.compile(individual, pset)
        img = GeneticImageArtifact.generate_image(func, shape)
        color_img = color_map[img]
        new_fname = "{}_{}x{}.png".format(fname[:-4], shape[0], shape[1])
        misc.imsave(new_fname, color_img)

    @staticmethod
    def save_artifact(artifact, folder, aid, pset, color_map,
                      shape=(400, 400), create_color=False):
        """
        Saves an artifact as .png.
        :param artifact:
            The artifact to be saved.
        :param folder:
            Path of the save folder.
        :param aid:
            Identification for the artifact.
        """
        s = artifact.framings['string_repr']
        try:
            individual = gp.PrimitiveTree.from_string(s, pset)
        except:
            print(traceback.format_exc())
            print(pset.__dict__)
            print(s)
            return
        func = gp.compile(individual, pset)
        img = GeneticImageArtifact.generate_image(func, shape)
        color_img = None
        if len(img.shape) == 2:
            if create_color:
                color_img = color_map[img]
        else:
            color_img = img
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        #print(color_img.shape, np.min(color_img), np.max(color_img))
        bname = "art{}".format(aid)
        if color_img is not None:
            imname = '{}.png'.format(bname)
            misc.imsave(os.path.join(folder, imname), color_img)
        imname = 'bw_{}.png'.format(bname)
        misc.imsave(os.path.join(folder, imname), img)

        fname = os.path.join(folder, 'f_{}.txt'.format(bname))
        with open(fname, 'w') as f:
            f.write("{}\n".format(s))

    @staticmethod
    def max_distance(create_kwargs, shape=None, bw=True):
        """Maximum distance between two images is calculated as the euclidean
        distance between an image filled with zeros and an image filled with
        255.
        """
        class DummyArtifact():
            def __init__(self, obj):
                self.obj = obj

        # TODO: This could be computed only once per shape!
        if shape is None:
            shape = create_kwargs['shape']
        if not bw and len(shape) < 3:
            shape = (shape[0], shape[1], 3)
        art1 = DummyArtifact(np.zeros(shape, dtype=np.uint8))
        art2 = DummyArtifact(np.zeros(shape, dtype=np.uint8) + 255)
        return GeneticImageArtifact.distance(art1, art2)

    @staticmethod
    def distance(artifact1, artifact2):
        """Euclidean distance between two artifact's which objects are images.

        Images are expected to be ``np.uint8``. The intensity values are
        converted to floats in [0, 1] before computing the distance.
        """
        im1 = artifact1.obj / 255.0
        im2 = artifact2.obj / 255.0
        if len(im1.shape) == 2:
            return np.sqrt(np.sum(np.square(im1 - im2)))
        else:
            distances = np.zeros(3)
            for i in range(3):
                ch1 = im1[:, :, i]
                ch2 = im2[:, :, i]
                distances[i] = np.sum(np.square(ch1 - ch2))
            return np.sqrt(np.sum(distances))

    @staticmethod
    def generate_image(func, shape=(32, 32), bw=True, str_func=None):
        """
        Creates an image.

        :param func:
            The function used to calculated color values.
        :param shape:
            Shape of the image.
        :return:
            A numpy array containing the color values.
            The format is uint8, because that is what opencv wants.
        """
        width = shape[0]
        height = shape[1]
        if bw:
            img = np.zeros(shape)
        else:
            img = np.zeros((shape[0], shape[1], 3))
        coords = [(x, y) for x in range(width) for y in range(height)]
        try:
            for x, y in coords:
                # Normalize coordinates in range [-1, 1]
                x_normalized = x / width * 2 - 1
                y_normalized = y / height * 2 - 1
                val = func(x_normalized, y_normalized)
                # TODO: is this going to work with RGB too if type(val) == list?
                if type(val) is not int:
                    val = np.around(val)
                img[x, y] = val
        except:
            print(traceback.format_exc())
            print(func, str_func)
            # Return black image if any errors occur.
            return np.zeros(img.shape, dtype=np.uint8)

        # Clip values in range [0, 255]
        img = np.clip(img, 0, 255, out=img)
        return np.uint8(img)

    @staticmethod
    def individual_to_artifact(individual, agent, shape):
        if individual.image is None:
            # If tree is too tall return negative evaluation
            try:
                func = gp.compile(individual, agent.super_pset)
            except MemoryError:
                return None
            func_str = str(individual)
            image = GeneticImageArtifact.generate_image(func, shape, func_str)
            individual.image = image

        # Convert deap individual to creamas artifact for evaluation
        artifact = GeneticImageArtifact(agent, individual.image, individual,
                                        str(individual))
        return artifact

    @staticmethod
    def evaluate(individual, agent, shape):
        """Evaluates a deap individual.

        :param individual:
            The individual to be evaluated.
        :param agent:
            The agent whose evaluation function is used.
        :param shape:
            Shape of the image.
        :return:
            The evaluation.
        """
        artifact = GeneticImageArtifact.individual_to_artifact(individual, agent, shape)

        if artifact is None:
            return -1,

        return agent.evaluate(artifact)

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
            Deap toolbox with the necessary functions.
        :param pset:
            The primitive set used in the evolution.
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
                offspring = tools.selBest(population, 1)
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
            ``deap.gp.genHalfAndHalf``.
        :return:
            A list containing the generated population.
        """
        GeneticImageArtifact.init_creator(pset)
        pop_toolbox = base.Toolbox()
        pop_toolbox.register("expr", generation_method, pset=pset, min_=2, max_=6)
        pop_toolbox.register("individual", tools.initIterate, creator.Individual,
                             pop_toolbox.expr)
        pop_toolbox.register("population", tools.initRepeat, list, pop_toolbox.individual)
        return pop_toolbox.population(size)

    @staticmethod
    def init_creator(pset):
        """Initializes the deap creator to use the wanted primitive set.
        """
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
                       pset=pset, image=None)

    '''
    @staticmethod
    def work_on_artifact(agent, artifact, create_kwargs, iterations=1):
        # if artifact is None:
        #     pop = GeneticImageArtifact.create_population(create_kwargs['pop_size'], create_kwargs['pset'])
        # else:
        #     pop = []
        #     for individual in artifact.framings['pop']:
        #         GeneticImageArtifact.init_creator(create_kwargs['pset'])
        #         pop.append(creator.Individual(individual))
        # GeneticImageArtifact.evolve_population(pop,
        #                                        iterations, create_kwargs['toolbox'],
        #                                        create_kwargs['pset'])
        # best = tools.selBest(pop, 1)[0]
        # new_artifact = GeneticImageArtifact(agent, best.image, list(best))
        # new_artifact.framings['pop'] = list(map(list, pop))

        toolbox = create_kwargs['toolbox']

        GeneticImageArtifact.init_creator(create_kwargs['pset'])
        ind1 = creator.Individual(artifact.framings['function_tree'])
        ind1.fitness.values = toolbox.evaluate(ind1)
        import copy

        # while True:
        #     mutant = copy.deepcopy(ind1)
        #     toolbox.mutate(mutant, create_kwargs['pset'])
        #     del mutant.fitness.values
        #     del mutant.image
        #     mutant.fitness.values = toolbox.evaluate(mutant)
        #     if mutant.fitness.values[0] > ind1.fitness.values[0]:
        #         ind1 = mutant
        #     if ind1.fitness.values[0] > 0.35:
        #         break

        ind2 = creator.Individual(agent.artifact.framings['function_tree'])
        ind2.fitness.values = toolbox.evaluate(ind2)

        i = 0
        while True:
            i += 1
            child1 = copy.deepcopy(ind1)
            child2 = copy.deepcopy(ind2)
            toolbox.mate(child1, child2)
            del child1.fitness.values
            del child2.fitness.values
            del child1.image
            del child2.image
            child1.fitness.values = toolbox.evaluate(child1)
            child2.fitness.values = toolbox.evaluate(child2)
            if child1.fitness.values[0] > ind1.fitness.values[0]:
                ind1 = child1
            if child2.fitness.values[0] > ind2.fitness.values[0]:
                ind2 = child2
            if ind1.fitness.values[0] > 0.4:
                break

        agent._log(logging.INFO, 'Mating iterations: ' + str(i))
        agent.artifact = GeneticImageArtifact(agent, ind2.image, list(ind2))

        return GeneticImageArtifact(agent, ind1.image, list(ind1))
    '''

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
                    individual = creator.SuperIndividual(
                        art.framings['function_tree'])
                    # Force mutate artifacts from the memory
                    if mutate_old:
                        toolbox.mutate(individual, pset)
                        del individual.fitness.values
                        if individual.image is not None:
                            del individual.image
                    population.append(individual)

            if len(population) < pop_size:
                population += GeneticImageArtifact.create_population(
                    pop_size - len(population), pset, gp.genHalfAndHalf)
            return population
        # Return random population
        return GeneticImageArtifact.create_population(pop_size, pset,
                                                      gp.genHalfAndHalf)

    @staticmethod
    def create(generations, agent, hall_of_fame, toolbox, pset, pop_size, shape,
               init_method='50-50'):
        """
        Creates an artifact.

        The best individuals created during the evolution are kept in the given
        hall of fame.

        :param generations:
            Number of generations evolved in the creation process.
        :param agent:
            The agent whose memory and evaluation function is used.
        :param hall_of_fame:
            deap's :class:`HallOfFame`-object to store the best individuals.
        :param toolbox:
            Deap toolbox used in the evolution.
        :param pset:
            Primitive set used in the evolution.
        :param pop_size:
            Size of the population.
        :param shape:
            Shape of the image.
        :param init_method:
            Population initialization method.
            See :meth:`GeneticImageArtifact.initial_population`.
        :return:
            The given :class:`HallOfFame`-object.
        """

        population = GeneticImageArtifact.initial_population(agent,
                                                             toolbox,
                                                             pset,
                                                             pop_size,
                                                             init_method)
        toolbox.register("evaluate", GeneticImageArtifact.evaluate,
                         agent=agent, shape=shape)
        GeneticImageArtifact.evolve_population(population, generations,
                                               toolbox, pset, hall_of_fame)
        return hall_of_fame

    @staticmethod
    def invent(n, agent, create_kwargs, n_artifacts=1):
        '''
        Invent new artifacts.

        :param n:
            Number of generations to be evolved.
        :param agent:
            The agent who is "creating" the artifact.
        :param create_kwargs:
            Parameters used in creating artifacts.
        :param n_artifacts:
            Number of artifacts to be created.
        :return:
            The invented artifact.
        '''
        hof = tools.HallOfFame(n_artifacts)
        hof = GeneticImageArtifact.create(n, agent, hof, **create_kwargs)
        arts = []
        for ft in hof:
            artifact = GeneticImageArtifact(agent, ft.image, list(ft), str(ft))
            arts.append((artifact, None))
        return arts

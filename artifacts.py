from creamas.core.artifact import Artifact

from deap import tools, creator, base
import deap.gp as gp

import logging
import numpy as np


class DummyArtifact(Artifact):
    def __init__(self, creator, obj):
        super().__init__(creator, obj, domain='dummy')

    @staticmethod
    def max_distance(create_kwargs):
        return np.linalg.norm(np.ones(create_kwargs['length']))

    @staticmethod
    def distance(artifact1, artifact2):
        obj1 = np.array(artifact1.obj)
        obj2 = np.array(artifact2.obj)
        return np.linalg.norm(obj1 - obj2)

    @staticmethod
    def create(length):
        return np.random.rand(length)

    @staticmethod
    def invent(n, agent, create_kwargs):
        obj = DummyArtifact.create(**create_kwargs)
        best_artifact = DummyArtifact(agent, obj)
        best_eval, _ = agent.evaluate(best_artifact)
        for _ in range(n - 1):
            obj = DummyArtifact.create(**create_kwargs)
            artifact = DummyArtifact(agent, obj)
            eval, _ = agent.evaluate(artifact)
            if eval > best_eval:
                best_artifact = artifact
                best_eval = eval
        best_artifact.add_eval(agent, best_eval)
        return best_artifact, None


class GeneticImageArtifact(Artifact):
    def __init__(self, creator, obj, function_tree):
        super().__init__(creator, obj, domain='image')
        self.framings['function_tree'] = function_tree

    @staticmethod
    def max_distance(create_kwargs):
        class DummyArtifact():
            def __init__(self, obj):
                self.obj = obj

        shape = create_kwargs['shape']
        art1 = DummyArtifact(np.zeros((shape[0], shape[1], 3)))
        art2 = DummyArtifact(np.ones((shape[0], shape[1], 3)) * 255)
        return GeneticImageArtifact.distance(art1, art2)

    @staticmethod
    def distance(artifact1, artifact2):
        im1 = artifact1.obj / 255
        im2 = artifact2.obj / 255
        distances = np.zeros(3)
        for i in range(3):
            ch1 = im1[:, :, i]
            ch2 = im2[:, :, i]
            distances[i] = np.sqrt(np.sum(np.square(ch1 - ch2)))
        return np.sqrt(np.sum(distances**2))

    @staticmethod
    def generate_image(func, shape=(32, 32)):
        width = shape[0]
        height = shape[1]
        image = np.zeros((width, height, 3))

        coords = [(x, y) for x in range(width) for y in range(height)]
        for coord in coords:
            x = coord[0]
            y = coord[1]
            x_normalized = x / width * 2 - 1
            y_normalized = y / height * 2 - 1
            color_value = np.around(np.array(func(x_normalized, y_normalized)))
            for i in range(3):
                if color_value[i] < 0:
                    image[x, y, i] = 0
                elif color_value[i] > 255:
                    image[x, y, i] = 255
                else:
                    image[x, y, i] = color_value[i]

        return np.uint8(image)

    @staticmethod
    def evaluate(individual, agent, shape):
        if individual.image is None:
            # If tree is too tall return negative evaluation
            try:
                func = gp.compile(individual, individual.pset)
            except MemoryError:
                return -1,
            image = GeneticImageArtifact.generate_image(func, shape)
            individual.image = image
        artifact = GeneticImageArtifact(agent, individual.image, individual)
        evaluation, _ = agent.evaluate(artifact)
        return evaluation,

    @staticmethod
    def evolve_population(population, generations, toolbox, pset):
        CXPB, MUTPB = 0.5, 0.2

        fitnesses = map(toolbox.evaluate, population)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        for g in range(generations):
            # Select the next generation individuals
            offspring = toolbox.select(population, len(population))
            # Clone the selected individuals
            offspring = list(map(toolbox.clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.random() < CXPB:
                    toolbox.mate(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values
                    del child1.image
                    del child2.image

            for mutant in offspring:
                if np.random.random() < MUTPB:
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

    @staticmethod
    def create_population(size, pset):
        GeneticImageArtifact.init_creator(pset)
        pop_toolbox = base.Toolbox()
        pop_toolbox.register("expr", gp.genHalfAndHalf, pset=pset, min_=2, max_=3)
        pop_toolbox.register("individual", tools.initIterate, creator.Individual,
                             pop_toolbox.expr)
        pop_toolbox.register("population", tools.initRepeat, list, pop_toolbox.individual)
        return pop_toolbox.population(size)

    @staticmethod
    def init_creator(pset):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", gp.PrimitiveTree, fitness=creator.FitnessMax,
                       pset=pset, image=None)

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

    @staticmethod
    def create(generations, agent, toolbox, pset, pop_size, shape):
        population = []

        if len(agent.stmem.artifacts) > 0:
            mem_size = min(pop_size, len(agent.stmem.artifacts))
            mem_arts = np.random.choice(agent.stmem.artifacts, size=mem_size, replace=False)
            for art in mem_arts:
                individual = creator.Individual(art.framings['function_tree'])
                population.append(individual)

        if len(population) < pop_size:
            population += GeneticImageArtifact.create_population(pop_size - len(population), pset)

        toolbox.register("evaluate", GeneticImageArtifact.evaluate, agent=agent, shape=shape)
        GeneticImageArtifact.evolve_population(population, generations, toolbox, pset)
        best = tools.selBest(population, 1)[0]
        return best


    @staticmethod
    def invent(n, agent, create_kwargs):
        function_tree = GeneticImageArtifact.create(n, agent, **create_kwargs)
        artifact = GeneticImageArtifact(agent, function_tree.image, list(function_tree))
        return artifact, None

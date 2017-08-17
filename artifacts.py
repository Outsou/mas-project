from creamas.core.artifact import Artifact

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

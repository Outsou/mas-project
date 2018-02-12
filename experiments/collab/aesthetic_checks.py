import logging
import pickle

import aiomas
import numpy as np
from creamas import Environment, Simulation
from creamas.rules import RuleLeaf
from creamas.mappers import DoubleLinearMapper

from artifacts import GeneticImageArtifact
from experiments.collab.base import DriftingGPCollaborationAgent, get_aid
from experiments.collab import collab_exp as coe

from features import ImageComplexityFeature, ImageEntropyFeature


class DriftTestAgent(DriftingGPCollaborationAgent):

    def __init__(self,
                 *args,
                 aesthetic='entropy',
                 aesthetic_bounds=[2.0, 5.5451],
                 aesthetic_target=3.0,
                 entropy_targets=None,
                 complexity_targets=None,
                 **kwargs):
        super_pset = coe.create_super_pset(bw=True)
        critic_threshold = coe.DEFAULT_PARAMS['critic_threshold']
        veto_threshold = coe.DEFAULT_PARAMS['veto_threshold']
        #novelty_weight = coe.DEFAULT_PARAMS['novelty_weight']
        novelty_weight = 0.5
        memsize = coe.DEFAULT_PARAMS['mem_size']
        search_width = coe.DEFAULT_PARAMS['search_width']
        shape = coe.DEFAULT_PARAMS['shape']
        # collab_model = coe.DEFAULT_PARAMS['model']  # Learning model
        collab_model = 'state-Q'
        output_shape = coe.DEFAULT_PARAMS['output_shape']
        aesthetics = [aesthetic]
        if aesthetic == 'entropy':
            feat = ImageEntropyFeature
        elif aesthetic == 'complexity':
            feat = ImageComplexityFeature
        dlm = DoubleLinearMapper(feat.MIN, aesthetic_target, feat.MAX, '01')
        rules = [RuleLeaf(feat(), dlm)]
        rule_weights = [1.0]
        create_kwargs, funnames = coe.get_create_kwargs(20, shape, 8)
        # print(create_kwargs['pset'])
        # Generated aesthetic values
        self.gen_aests = []
        self.imf = ImageComplexityFeature()
        self.ief = ImageEntropyFeature()
        self.complexity_targets = complexity_targets
        self.entropy_targets = entropy_targets

        self.imf_mappers = {t: DoubleLinearMapper(ImageComplexityFeature.MIN, t,
                                                  ImageComplexityFeature.MAX,
                                                  '01') for t in
                            complexity_targets}
        self.ief_mappers = {t: DoubleLinearMapper(ImageEntropyFeature.MIN, t,
                                                  ImageEntropyFeature.MAX,
                                                  '01') for t in
                            entropy_targets}

        self.mapped_values = {
            'entropy': {t: {e: [] for e in complexity_targets} for t in entropy_targets},
            'complexity': {t: {e: [] for e in entropy_targets} for t in complexity_targets},
        }

        super().__init__(
            *args,
            save_folder='drift_test',
            log_folder='drift_test',
            aesthetic_target=aesthetic_target,
            aesthetic_bounds=aesthetic_bounds,
            create_kwargs=create_kwargs,
            artifact_cls=GeneticImageArtifact,
            rules=rules,
            rule_weights=rule_weights,
            memsize=memsize,
            critic_threshold=critic_threshold,
            veto_threshold=veto_threshold,
            novelty_weight=novelty_weight,
            search_width=search_width,
            output_shape=output_shape,
            collab_model=collab_model,
            super_pset=super_pset,
            aesthetic=aesthetics[0],
            novelty_threshold=0.01,
            value_threshold=0.01,
            pset_names=funnames,
            drifting_prob=0.0,
            **kwargs)

    def _get_aest_value(self, artifact):
        if self.aesthetic == 'entropy':
            return self.ief.extract(artifact)
        elif self.aesthetic == 'complexity':
            return self.imf.extract(artifact)
        return 0.0

    def _evaluate_with(self, artifact, feat, mapper):
        return mapper(feat.extract(artifact))

    def _map_value(self, artifact):
        if self.aesthetic == 'entropy':
            for imf_target, mapper in self.imf_mappers.items():
                value = self._evaluate_with(artifact, self.imf, mapper)
                self.mapped_values['entropy'][self.aesthetic_target][imf_target].append(value)
                print("Entropy {} mapped for complexity {}: {}".format(
                    self.aesthetic_target, imf_target, value))

        if self.aesthetic == 'complexity':
            for ief_target, mapper in self.ief_mappers.items():
                value = self._evaluate_with(artifact, self.ief, mapper)
                self.mapped_values['complexity'][self.aesthetic_target][ief_target].append(value)
                print("Complexity {} mapped for entropy {}: {}".format(
                    self.aesthetic_target, ief_target, value))



    @aiomas.expose
    async def act(self, *args, **kwargs):
        self.age += 1
        artifacts = self.invent(self.search_width, n_artifacts=1)
        artifact = artifacts[0][0]

        obj_value = self._get_aest_value(artifact)
        self._map_value(artifact)

        e, fr = self.evaluate(artifact)
        n = None if fr['novelty'] is None else np.around(fr['novelty'], 2)
        v = np.around(fr['value'], 2)
        self._log(logging.INFO,
                  'Created an individual artifact. E={} (V:{}, N:{})'
                  .format(np.around(e, 2), v, n))
        print("E:{} V:{} N:{} O:{} T:{}"
              .format(e, fr['value'], fr['novelty'], obj_value,
                      self.aesthetic_target))
        self.learn(artifact)
        aid = get_aid(self.addr, self.age, self.aesthetic, v, n)
        artifact.aid = aid
        #self.save_artifact(artifact, pset=self.super_pset, aid=aid)s

    def save_mapped_values(self):
        with open('mapped_values.pkl', 'wb') as f:
            pickle.dump(self.mapped_values, f)


def aesthetic_ch():
    aesthetic = 'entropy'
    entropy_bounds = [0.499, 5.001]
    entropy_targets = [0.50 + (0.25*i) for i in range(19)]
    #entropy_targets = [0.50 + (0.25 * i) for i in range(3)]
    complexity_bounds = [0.499, 2.301]
    complexity_targets = [0.5 + (0.1 * i) for i in range(19)]
    #complexity_targets = [0.5 + (0.1 * i) for i in range(3)]
    artifacts_per_target = 20

    ce = Environment.create(('localhost', 5555))
    da = DriftTestAgent(ce,
                        aesthetic=aesthetic,
                        aesthetic_bounds=entropy_bounds,
                        aesthetic_target=1.0,
                        entropy_targets=entropy_targets,
                        complexity_targets=complexity_targets)

    for target in entropy_targets:
        print("Entropy target {} ({} artifacts)".format(target, artifacts_per_target))
        da.aesthetic_target = target
        for i in range(artifacts_per_target):
            aiomas.run(da.act())

    da.aesthetic_bounds = complexity_bounds
    da.aesthetic = 'complexity'
    for target in complexity_targets:
        print("Complexity target {} ({} artifacts)".format(target, artifacts_per_target))
        da.aesthetic_target = target
        for i in range(artifacts_per_target):
            aiomas.run(da.act())

    print("Finished all targets.")
    da.save_mapped_values()
    ce.destroy()


def main(aesthetic='entropy',
         aesthetic_bounds=[0.5, 5.0],
         aesthetic_target=0.5001):
    ce = Environment.create(('localhost', 5555))
    da = DriftTestAgent(ce,
                        aesthetic=aesthetic,
                        aesthetic_bounds=aesthetic_bounds,
                        aesthetic_target=aesthetic_target)
    steps = 1000
    print("Starting {} steps".format(steps))
    for i in range(steps):
        print("Step {}".format(i+1))
        if i % 100 == 0:
            print("Changing agent target!")
            da.aesthetic_target = 5.0
        aiomas.run(da.act())
    print("Finished steps.")
    ce.destroy()


if __name__ == "__main__":
    aesthetic_ch()

from creamas.rules.feature import Feature


class DummyFeature(Feature):
    def __init__(self, feature_idx):
        super().__init__('dummy', ['dummy'], float)
        self.feature_idx = feature_idx

    def extract(self, artifact):
        return float(artifact.obj[self.feature_idx])
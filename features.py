from creamas.rules.feature import Feature


class DummyFeature(Feature):
    '''A dummy feature used for testing purposes.'''
    def __init__(self, feature_idx):
        '''
        :param feature_idx:
            The index of the feature that will be extracted.
        '''
        super().__init__('dummy', ['dummy'], float)
        self.feature_idx = feature_idx

    def extract(self, artifact):
        return float(artifact.obj[self.feature_idx])
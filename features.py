from creamas.rules.feature import Feature

import numpy as np
import cv2

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


class ImageBenfordsLawFeature(Feature):
    """Feature computing the Benford's Law for images.

    .. seealso::
        `Benford's Law<https://en.wikipedia.org/wiki/Benford%27s_law>`_
    """
    def __init__(self, ):
        super().__init__("image_Benfords_law", ['image'], float)
        # Benford's histogram's bin values
        self.b = [0.301, 0.176, 0.125, 0.097, 0.079, 0.067, 0.058, 0.051, 0.046]
        self.b_max = (1.0 - self.b[0]) + np.sum(self.b[1:])

    def extract(self, artifact):
        img = artifact.obj
        # Convert color image to black and white
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hist = cv2.calcHist([img], [0], None, [9], [0, 256])
        # Sort, reverse and rescale to give the histogram a sum of 1.0
        h2 = np.sort(hist, 0)[::-1] * (1.0 / np.sum(hist))
        # Compute Benford's Law feature
        total = np.sum([h2[i] - self.b[i] for i in range(len(h2))])
        return 1.0 - (total / self.b_max)


class ImageColorfulnessFeature(Feature):
    """Compute image's colorfulness.

    Accepts only RBG color images.
    """
    def __init__(self):
        super().__init__('image_colourfulness', ['image'], float)

    def extract(self, artifact):
        img = artifact.obj
        # Colorfulness is only applicable to color images.
        if len(img.shape) < 3:
            return 0.0
        if img.dtype != np.float:
            img = img / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        delta_rg, delta_yb = np.abs(r-b), np.abs((r+b)/2.0 - b)
        mrg, myb = np.mean(delta_rg), np.mean(delta_yb)
        srg, syb = np.std(delta_rg), np.std(delta_yb)
        return np.sqrt(srg**2 + syb**2) + 0.3 * np.sqrt(mrg**2 + myb**2)


class ImageMeanHueFeature(Feature):
    """Compute mean hue for the image in HSV color space.

    TODO: Currently this is badly implemented (but as described in the paper),
    because hues close to 0 and 1 are close to each other but cause the mean
    hue diverge.
    """
    def __init__(self):
        super().__init__('image_mean_hue', ['image'], float)

    def extract(self, artifact):
        img = artifact.obj
        # Mean hue is only applicable to color images.
        if len(img.shape) < 3:
            return 0.0
        if img.dtype != np.float:
            img = img / 255.0
        hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        h = hsv_img[:, :, 0]
        return np.mean(h)
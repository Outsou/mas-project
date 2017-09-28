from creamas.rules.feature import Feature

import numpy as np
import cv2


def fractal_dimension(image):
    '''Estimates the fractal dimension of an image with box counting.
    Counts pixels with value 0 as empty and everything else as non-empty.
    Input image has to be grayscale.

    See, e.g `Wikipedia <https://en.wikipedia.org/wiki/Fractal_dimension>`_.

    :param image: numpy.ndarray
    :returns: estimation of fractal dimension
    :rtype: float
    '''
    pixels = np.asarray(np.nonzero(image > 0)).transpose()
    lx = image.shape[1]
    ly = image.shape[0]
    if len(pixels) < 2:
        return 0
    scales = np.logspace(1, 8, num=20, endpoint=False, base=2)
    scales = scales[scales < image.shape[0] & image.shape[1]]
    Ns = []
    for scale in scales:
        H, edges = np.histogramdd(pixels,
                                  bins=(np.arange(0, lx, scale),
                                        np.arange(0, ly, scale)))
        H_sum = np.sum(H > 0)
        if H_sum == 0:
            H_sum = 1
        Ns.append(H_sum)

    coeffs = np.polyfit(np.log(scales), np.log(Ns), 1)
    hausdorff_dim = -coeffs[0]

    return hausdorff_dim


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

    def __str__(self):
        return '{}_{}'.format(self.name, self.feature_idx)


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
        total = np.sum([np.abs(h2[i] - self.b[i]) for i in range(len(h2))])
        benford = float(1.0 - (total / self.b_max))
        return 0.0 if benford < 0 else benford


class ImageColorfulnessFeature(Feature):
    """Compute image's colorfulness.

    Accepts only RBG color images.

    (This is not very good aesthetic measure.)
    """
    def __init__(self):
        super().__init__('image_colorfulness', ['image'], float)

    def extract(self, artifact):
        img = artifact.obj
        # Colorfulness is only applicable to color images.
        if len(img.shape) < 3:
            return 0.0
        if img.dtype != np.float:
            img = img / 255.0
        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        delta_rg, delta_yb = np.abs(r-g), np.abs((r+g)/2.0 - b)
        mrg, myb = np.mean(delta_rg), np.mean(delta_yb)
        srg, syb = np.std(delta_rg), np.std(delta_yb)
        cf = float(np.sqrt(srg**2 + syb**2) + 0.3 * np.sqrt(mrg**2 + myb**2))
        return cf


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
        return float(np.mean(h))


class ImageEntropyFeature(Feature):
    def __init__(self):
        super().__init__('image_entropy', ['image'], float)
        # Max entropy for 256 bins, i.e. the histogram has even distribution
        self.max = 5.5451774444795623

    def extract(self, artifact):
        img = artifact.obj
        # Convert color image to black and white
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        hg = cv2.calcHist([img], [0], None, [256], [0, 256])
        # Compute probabilities for each bin in histogram
        hg = hg / (img.shape[0] * img.shape[1])
        # Compute entropy based on bin probabilities
        e = -np.sum([hg[i] * np.log(hg[i]) for i in range(len(hg)) if hg[i] > 0.0])
        return float(e) / self.max


class ImageComplexityFeature(Feature):
    """Feature that estimates the fractal dimension of an image.
    The color values must be in range [0, 255] and type ``int``.
    """
    def __init__(self):
        super().__init__('image_complexity', ['image'], float)

    def extract(self, artifact):
        img = artifact.obj
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img, 100, 200)
        return float(fractal_dimension(edges))


class ImageFDAestheticsFeature(Feature):
    """Computes aesthetics from the fractal dimension. The value of the image
    is higher the closer the image's fractal dimension is to 1.35. The actual
    value function is ``max(0, 1 - |1.35 - fd(I)|)``.
    """
    def __init__(self):
        super().__init__('image_complexity', ['image'], float)

    def extract(self, artifact):
        img = artifact.obj
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(img, 100, 200)
        return max(0.0, 1 - abs(1.35 - float(fractal_dimension(edges))))


class ImageSymmetryFeature(Feature):
    """Compute symmetry of the image in given axis.
    """
    HORIZONTAL = 1
    VERTICAL = 2
    DIAGONAL = 4

    def __init__(self, axis, use_entropy=True):
        super().__init__('image_symmetry', ['image'], float)
        self.axis = axis
        self.threshold = 13
        b = "{:0>3b}".format(self.axis)
        self.horizontal = int(b[2])
        self.vertical = int(b[1])
        self.diagonal = int(b[0])
        self.liveliness = use_entropy

    def _hsymm(self, left, right):
        fright = np.fliplr(right)
        delta = np.abs(left-fright)
        t = delta <= self.threshold
        sim = np.sum(t) / (left.shape[0] * left.shape[1])
        return sim

    def _vsymm(self, up, down):
        fdown = np.flipud(down)
        delta = np.abs(up-fdown)
        t = delta <= self.threshold
        sim = np.sum(t) / (up.shape[0] * up.shape[1])
        return sim

    def _dsymm(self, ul, ur, dl, dr):
        fdr = np.fliplr(np.flipud(dr))
        fur = np.fliplr(np.flipud(ur))
        d1 = np.abs(ul - fdr)
        d2 = np.abs(dl - fur)
        t1 = d1 <= self.threshold
        t2 = d2 <= self.threshold
        s1 = np.sum(t1) / (ul.shape[0] * ul.shape[1])
        s2 = np.sum(t2) / (ul.shape[0] * ul.shape[1])
        return (s1 + s2) / 2

    def extract(self, artifact):
        img = artifact.obj
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        cx = int(img.shape[0] / 2)
        cy = int(img.shape[1] / 2)
        n = 0
        symms = 0.0
        liv = 1.0
        if self.horizontal:
            symms += self._hsymm(img[:, :cx], img[:, cx:])
            n += 1
        if self.vertical:
            symms += self._vsymm(img[:cy, :], img[cy:, :])
            n += 1
        if self.diagonal:
            symms += self._dsymm(img[:cy, :cx], img[:cy, cx:],
                                 img[cy:, :cx], img[cy:, cx:])
            n += 1
        if self.liveliness:
            ie = ImageEntropyFeature()
            liv = ie(artifact)

        return float(liv * (symms / n))



"""
class ImageCascadeClassifierFeature(Feature):
    def __init__(self, cascade):
        super().__init__('image_cascade_classifier', ['image'], float)
        self.cascade = cascade

    def extract(self, artifact):
        img = artifact.obj
        # Convert color image to black and white
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        objs, rej_levelvs, level_weights = self.cascade.detectMultiScale3(img, outputRejectLevels=True)
        if len(objs) == 0:
            return 0.0
        return len(objs)
"""

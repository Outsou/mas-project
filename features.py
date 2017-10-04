from functools import reduce
import time

from creamas.rules.feature import Feature
from scipy.stats import norm
from scipy import misc
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
    ns = []
    vs = []
    for scale in scales:
        x_bins = np.arange(0, lx, scale)
        x_bins = np.concatenate((x_bins, [lx])) if x_bins[-1] < lx else x_bins
        y_bins = np.arange(0, ly, scale)
        y_bins = np.concatenate((y_bins, [ly])) if y_bins[-1] < ly else y_bins
        H, edges = np.histogramdd(pixels, bins=(x_bins, y_bins))
        H_sum = np.sum(H > 0)
        if H_sum > 0:
            ns.append(H_sum)
            vs.append(scale)

    coeffs = np.polyfit(np.log(vs), np.log(ns), 1)
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
    """Compute entropy of an image and normalize it to interval [0, 1].

    Entropy computation uses 256 bins and a grey scale image.
    """
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
    """Computes aesthetics from the fractal dimension.

    The value of the feature is higher the closer the image's fractal dimension
    is to 1.35. The precise value function is ``max(0, 1 - |1.35 - fd(I)|)``.
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
    """Compute symmetry of the image in given ax or combination of axis.

    Feature also allows adding the computed symmetry with "liveliness" of the
    image using ``use_entropy=True``. If entropy is not used, simple images
    (e.g. plain color images) will give high symmetry values.

    :param axis:
        :attr:`ImageSymmetryFeature.HORIZONTAL`,
        :attr:`ImageSymmetryFeature.VERTICAL`, and/or
        :attr:`ImageSymmetryFeature.DIAGONAL`.

        These can be combined, e.g. ``axis=ImageSymmetryFeature.HORIZONTAL+
        ImageSymmetryFeature.VERTICAL``.

    :param bool use_entropy:
        If ``True`` multiples the computed symmetry value with image's entropy
        ("liveliness").
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


class ImageBellCurveFeature(Feature):
    """Compute Ross, Ralph and Zong image aesthetic measure, also known as
    bell curve measure.
    """
    def __init__(self):
        super().__init__('image_bell_curve', ['image'], float)

    def _gradient(self, ch):
        w, h = ch.shape
        g = np.zeros((w - 1, h - 1), dtype=np.float64)
        #d = 0.1 * np.sqrt((w**2 + h**2)) / 2.0
        d = 1.0
        for x in range(w - 1):
            for y in range(h - 1):
                g1 = np.float_power(float(ch[x][y]) - float(ch[x + 1][y + 1]), 2)
                g2 = np.float_power(float(ch[x][y + 1]) - float(ch[x + 1][y]), 2)
                g[x][y] = (g1 + g2) / d ** 2
                #print(x, y, g1, g2, ch[x][y], ch[x + 1][y + 1], ch[x + 1][y],
                #      ch[x][y + 1], g[x][y])
        return g

    def _combine_gradients(self, grs):
        total_gradient = reduce(lambda x, y: x + y, grs)
        return np.sqrt(total_gradient)

    def _dfn(self, hg, bins, mean, std):
        # Gaussian distribution's cdf's in bin intersections
        cdfs = norm.cdf(bins, loc=mean, scale=std)
        # Gaussian distribution's probability for each bin
        qi = np.array([cdfs[i] - cdfs[i - 1] for i in range(1, len(cdfs))])

        # a lot of filtering so that taking a logarithm does not mess things
        hg_filtered = hg[qi != 0]
        qi_filtered = qi[qi != 0]
        pq = hg_filtered / qi_filtered
        pq_filtered = pq[pq != 0]
        hg_filtered = hg_filtered[pq != 0]
        return np.sum(hg_filtered * np.log(pq_filtered))

    def extract(self, artifact):
        img = artifact.obj
        if len(img.shape) == 2:
            s = self._combine_gradients([self._gradient(img)])
        else:
            grs = [self._gradient(img[:, :, i]) for i in range(3)]
            s = self._combine_gradients(grs)
        # Here s0 = 2 is used for both gray and color images. However, it is
        # not exactly clear if it should be 2 for gray images.
        r = s / 2.0
        rsum = np.sum(r)
        mean = np.sum(np.float_power(r, 2)) / rsum
        #std = np.sqrt(np.sum(np.float_power(r - mean, 2)) / (r.shape[0] * r.shape[1]))
        std = np.sum(r * ((r - mean) ** 2)) / rsum
        maxr = np.max(r)
        s100 = np.sqrt(std) / 100
        bins = np.concatenate((np.arange(0, maxr, s100), [maxr]))
        # Density histogram of the bins
        hg = np.histogram(r, bins, weights=r)[0]
        dhg = hg / np.sum(hg)
        dfn = float(self._dfn(dhg, bins, mean, std))
        print("DFN: {}, mean: {}, std: {}".format(dfn, mean, std))
        return dfn / 100


class ImageGlobalContrastFactorFeature(Feature):
    """Compute global contrast factor from an image.
    """
    def __init__(self):
        super().__init__('image_global_contrast_factor', ['image'], float)
        self.gamma = 2.2
        self.m1 = -0.406385   # Magic number 1
        self.m2 = 0.334573   # Magic number 2
        self.m3 = 0.0877526  # Magic number 3
        self.sps = [1, 2, 4, 8, 16, 25, 50, 100, 200]  # Super pixel sizes

    def m_func(self, x):
        return (self.m1 * x + self.m2) * (x + self.m3)

    def resize(self, img, scale):
        """Compute a new image with super pixels of a given scale.

        Returns an image consisting of super pixels.
        """
        if scale == 1:
            return img
        bins = [e for e in range(scale, img.shape[0], scale)]
        # Split image to super pixels
        split_img = np.split(img, bins)
        for i,e in enumerate(split_img):
            split_img[i] = np.split(e, bins, axis=-1)
        sp_img = np.zeros((len(split_img), len(split_img[0])))
        # Compute super pixel values as the average of luminances
        for x in range(len(split_img)):
            for y in range(len(split_img[0])):
                n_pixels = split_img[x][y].shape[0] * split_img[x][y].shape[1]
                sp_img[x, y] = np.sum(split_img[x][y]) / n_pixels
        return sp_img

    def contrast(self, img, scale):
        """Compute contrast on a given superpixel scale.
        """
        sp_img = self.resize(img, scale)
        sum_contrast = 0
        for x in range(sp_img.shape[0]):
            for y in range(sp_img.shape[1]):
                px = sp_img[x, y]
                l = 0.0
                d = 0
                if x > 0:
                    l += abs(px - sp_img[x - 1][y])
                    d += 1
                if y > 0:
                    l += abs(px - sp_img[x][y - 1])
                    d += 1
                if x < sp_img.shape[0] - 1:
                    l += abs(px - sp_img[x + 1][y])
                    d += 1
                if y < sp_img.shape[1] - 1:
                    l += abs(px - sp_img[x][y + 1])
                    d += 1
                local_contrast = l / d
                sum_contrast += local_contrast
        avg_contrast = sum_contrast / (sp_img.shape[0] * sp_img.shape[1])
        return avg_contrast

    def extract(self, artifact):
        img = artifact.obj
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        # Gamma correction
        gamma_img = (img / 255.0) ** 2.2
        # Perceptual luminance
        pl_img = 100 * np.sqrt(gamma_img)
        scales = [e for e in self.sps if e < img.shape[0] and e < img.shape[1]]
        # TODO: use smaller scales as intermediate steps to compute larger.
        contrasts = [self.contrast(pl_img, scale) for scale in scales]
        #print(contrasts)
        wc = [self.m_func(i/len(scales)) * c for i, c in enumerate(contrasts)]
        #print(wc)
        gcf = np.sum(wc)
        #print(gcf)
        return float(gcf) / 10.0


class ImageMCFeature(Feature):
    """Compute aesthetic value by Machado & Cardoso for an image.

    Aesthetic value is computed as :math:`IC^a / PC^b`, where IC is the
    estimated image complexity and PC is the estimated processing complexity.

    IC is estimated with: :math`(s(c(i)) / s(i)) * RSME(i, c(i))`, where c(i)
    is the image encoded with jpeg compression, s is the file size function and
    RSME is the root mean squared error between the compressed and the original
    image.

    PC is estimated by compressing the image with jpeg2000 standard.
    """
    def __init__(self, a=1.0, b=1.0):
        """

        :param a:
        :param b:
        """
        super().__init__('image_machado_cardoso', ['image'], float)
        self.a = a
        self.b = b

    def jpeg(self, img, quality=75):
        _, buf = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        jpeg_img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return jpeg_img, len(buf)

    def jpeg2000(self, img):
        _, buf = cv2.imencode('.jp2', img)
        jpeg_img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return jpeg_img, len(buf)

    def png(self, img, comp=1):
        _, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, comp])
        png_img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        return png_img, len(buf)

    def rmse(self, img1, img2):
        n = img1.shape[0] * img1.shape[1]
        delta_img = (img1 * 1.0) - (img2 * 1.0)
        return np.sqrt(np.sum(delta_img ** 2) / n)

    def image_complexity(self, img):
        jpeg_img, jpeg_size = self.jpeg(img)
        #print(jpeg_img.shape, jpeg_img.dtype)
        # .bmp size
        original_size = (img.shape[0] * img.shape[1]) + 1078
        if len(img.shape) == 3:
            original_size = (img.shape[0] * img.shape[1] * img.shape[2]) + 54
        ratio = jpeg_size / original_size
        compression_error = self.rmse(img, jpeg_img)
        return compression_error * ratio

    def processing_complexity(self, img):
        # Now resulting to this naive way of computing processing complexity
        jp2, t0 = self.jpeg2000(img)
        original_size = (img.shape[0] * img.shape[1]) + 1078
        t0_ratio = t0 / original_size
        return t0_ratio
        '''
        jp2, t0 = self.jpeg2000(img)
        original_size = (img.shape[0] * img.shape[1]) + 1078
        t0_ratio = t0 / original_size
        t0_compression_error = self.rmse(jp2, img)

        cx = int(img.shape[0] / 2)
        cy = int(img.shape[1] / 2)
        qs = [img[:cx, :cy], img[cx:, :cy], img[:cx, cy:], img[cx:, cy:]]
        qs = [misc.imresize(q, img.shape, interp='nearest') for q in qs]
        jps = []
        t1 = 0.0
        for q in qs:
            cur_img, td = self.jpeg2000(q)
            print(td, 400*400+1078)
            jps.append((cur_img, q))
            t1 += td
        original_size2 = (img.shape[0] * img.shape[1] * 4) + 1078
        t1_ratio = t1 / original_size2
        t1_compression_error = 0.0
        for cur_img, q in jps:
            t1_compression_error += self.rmse(cur_img, q)
            print("RMSE: {}".format(t1_compression_error))
        t1_compression_error = t1_compression_error / 4
        pc_t0 = t0_ratio * t0_compression_error
        pc_t1 = t1_ratio * t1_compression_error
        print(t0_ratio, t0_compression_error, pc_t0, t1_ratio, t1_compression_error, pc_t1)
        pc = ((pc_t0 * pc_t1) ** 0.4) * ((pc_t1 - pc_t0) / pc_t1) ** 0.2
        print(pc)
        return pc
        '''

    def extract(self, artifact):
        img = artifact.obj
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        ic = self.image_complexity(img)
        #print("IC: {}".format(ic))
        pc = self.processing_complexity(img)
        #print(pc)
        return float(ic / pc)



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

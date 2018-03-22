"""A module holding an artifact implementation for the images created using genetic programming,
:class:`GPImageArtifact`.

The artifact implementation is coupled with :class:`GPImageGenerator`, the generator producing
evolutionary art using genetic programming.

.. info::

    This module needs numpy, scipy, deap and opencv2 installed.
"""
import numpy as np
import scipy as sp
import deap.gp
import cv2

from creamas.core.artifact import Artifact

from gp.generator import GPImageGenerator


class GPImageArtifact(Artifact):
    """A class for artifacts generated using :class:`GPImageGenerator`.

    Each :attr:`GPImageArtifact.obj` is a numpy array with a data type ``np.uint8``.

    """
    def __init__(self, creator, obj, function_tree, string_repr=None):
        """
        :param creator:
            Name of the creator agent.
        :param obj:
            Image generated, numpy array with ``dtype==np.uint8``.
        :param function_tree:
            Function from which the image was generated. This is stored to
            `framings['function_tree']`.
        :param string_repr:
            String representation of the function. This is stored to `framings['string_repr']`.
        """
        super().__init__(creator, obj, domain='image')
        self.framings['function_tree'] = function_tree
        self.framings['string_repr'] = string_repr
        self.png_compression_done = False
        # Artifact ID #
        self.aid = None
        self.rank = None
        self._feat_vals = {}    # Objective feature values for each feature.

    def add_feature_value(self, feat, val):
        """Add objective feature value for given feature.
        """
        self._feat_vals[feat] = val

    def get_feature_value(self, feat):
        """Return objective feature value for given feature, or ``None`` if it is not found.
        """
        if feat in self._feat_vals:
            return self._feat_vals[feat]
        return None

    @staticmethod
    def _individual_from_file(fname, pset):
        """Recreate an individual from a string saved into a file.
        """
        s = ""
        with open(fname, 'r') as f:
            s = f.readline()
        s = s.strip()
        individual = gp.PrimitiveTree.from_string(s, pset)
        return individual

    @staticmethod
    def image_from_file(individual_file, image_file, pset, color_map=None, shape=(400, 400)):
        """Save an individual saved as a string into a file as a new image with given
        color mapping and resolution.

        The function uses :func:`scipy.misc.imsave`.

        :param str individual_file:
            Path to the file with the string representation of the individual.
        :param str image_file:
            Path to the file where image is saved. The image type is defined by the file type.
        :param pset:
            DEAP's primitive set required to compile string representation of the individual into an
            individual.
        :param color_map:
            Color map used to colorize a greyscale image, e.g. one of the matplotlib's color maps.
        :param tuple shape:
            Shape of the generated image. Default is (400, 400).
        """
        individual = GPImageArtifact._individual_from_file(individual_file, pset)
        func = deap.gp.compile(individual, pset)
        img = GPImageArtifact.func2image(func, shape)
        if color_map is not None and len(img.shape) == 2:
            color_img = color_map[img]
        else:
            color_img = img
        sp.misc.imsave(image_file, color_img)

    @staticmethod
    def save(artifact, image_file, pset, color_map=None, shape=(400, 400), string_file=None):
        """
        Saves an artifact as an image.

        :param artifact:
            The artifact to be saved.
        :param str image_file:
            Path to the file where image is saved. The image type is defined by the file type.
        :param pset:
            DEAP's primitive set required to compile string representation of the individual into an
            individual.
        :param color_map:
            Color map used to colorize a greyscale image, e.g. one of the matplotlib's color maps.
        :param tuple shape:
            Shape of the generated image. Default is (400, 400).
        :param str save_string:
            If not ``None`` saves also the function associated with the artifact as txt in to the
            given file.
        """
        s = artifact.framings['string_repr']
        individual = deap.gp.PrimitiveTree.from_string(s, pset)
        func = deap.gp.compile(individual, pset)
        img = GPImageGenerator.func2image(func, shape)
        img = None
        if color_map is not None and len(img.shape) == 2:
            img = color_map[img]
        else:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        sp.misc.imsave(image_file, img)

        if string_file is not None:
            with open(string_file, 'w') as f:
                f.write("{}\n".format(s))

    @staticmethod
    def func2image(func, shape=(32, 32), bw=True):
        """Generate image from the given function.

        :param func:
            A function returned by :func:`deap.gp.compile` used to compute the color values.
        :param shape:
            Shape of the returned image.
        :param bool bw:
            If ``True``, ``func`` is assumed to represent RGB image, otherwise it is assumed to
            be a greyscale image.
        :return:
            A numpy array containing the color values.
            The format is uint8, because that is what opencv wants.

            If any errors occur during the image creation, fails silently and returns a black image.
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
                x_normalized = x / (width - 1) * 2 - 1
                y_normalized = y / (height - 1) * 2 - 1
                val = func(x_normalized, y_normalized)
                # TODO: is this going to work with RGB too if type(val) == list?
                if type(val) is not int:
                    val = np.around(val)
                img[x, y] = val
        except:
            # Return black image if any errors occur.
            return np.zeros(img.shape, dtype=np.uint8)

        # Clip values in range [0, 255]
        img = np.clip(img, 0, 255, out=img)
        return np.uint8(img)

    @staticmethod
    def max_distance(shape, bw=True):
        """Maximum distance between two images is calculated as the euclidean
        distance between an image filled with zeros and an image filled with
        255.
        """
        class DummyArtifact():
            def __init__(self, obj):
                self.obj = obj

        # TODO: This could be computed only once per shape!
        if not bw and len(shape) < 3:
            shape = (shape[0], shape[1], 3)
        art1 = DummyArtifact(np.zeros(shape, dtype=np.uint8))
        art2 = DummyArtifact(np.zeros(shape, dtype=np.uint8) + 255)
        return GPImageArtifact.distance(art1, art2)

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

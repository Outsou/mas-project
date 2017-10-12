"""
Helper functions to compute the winner from two Hall of Fames of GP images
"""
import operator
import random

import numpy as np


def rmse(img1, img2):
    """Compute root mean square error between two images of same shape.
    """
    n = img1.shape[0] * img1.shape[1]
    delta_img = (img1 * 1.0) - (img2 * 1.0)
    return np.sqrt(np.sum(delta_img ** 2) / n)


def filter(hof, epsilon=0.02):
    """Filter given ordered list of artifacts for distinct images using RMSE.

    :param hof: Hall of fame of images
    :param float epsilon:
        Threshold value for RMSE for two images to be seen as the same image.
    :returns
        A list of distinct images filtered and their ranks
    """
    # Put best artifact into the filtered list by default
    filtered = [(hof[0][0], 1)]
    # Filter all other artifacts against a set of kept artifacts.
    for i in range(1, len(hof)):
        obj = hof[i][0].obj
        passed = True
        for f, r in filtered:
            err = rmse(obj, f.obj)
            if err < epsilon:
                #print("Filtered an image with rmse={}".format(err))
                passed = False
                break
        if passed:
            #print("Appending an image with rank {}".format(i + 1))
            filtered.append((hof[i][0], i + 1))
    print("{} artifacts passed filtering.".format(len(filtered)))
    return filtered


def match(fl1, fl2, epsilon=0.02):
    """Match two filtered lists of images using epsilon as threshold value.

    :param fl1:
        A filtered lists of accepted artifacts and their ranks.
    :param fl2:
        A filtered lists of accepted artifacts and their ranks.
    :param float epsilon:
        Threshold value for RMSE for two images to be seen as the same image.
    """
    matched = []
    for a1, rank1 in fl1:
        obj1 = a1.obj
        for a2, rank2 in fl2:
            obj2 = a2.obj
            if rmse(obj1, obj2) < epsilon:
                matched.append((a1, a2, rank1 + rank2))
                # Always choose the best single match for an artifact
                break

    matched = sorted(matched, key=operator.itemgetter(2), reverse=False)
    return matched


def choose_best(hof1, hof2, epsilon=0.02):
    """Choose single best artifact from two ordered lists of artifacts using
    sum of ranks as the key.

    Lists are first filtered and then matched against each other.
    """
    fl1 = filter(hof1, epsilon)
    fl2 = filter(hof2, epsilon)
    matches = match(fl1, fl2, epsilon)
    a1, a2 = fl1[0][0].creator, fl2[0][0].creator
    print("{} and {} matched {} artifacts.".format(a1, a2, len(matches)))
    if len(matches) == 0 or matches is None:
        return None, None
    return random.choice((matches[0][0], matches[0][1])), matches[0][2]







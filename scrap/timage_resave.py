import os

import numpy as np
from matplotlib import cm

from utils import util
from artifacts import GeneticImageArtifact as GIA

from features import ImageGlobalContrastFactorFeature as IGCFF
from features import ImageMCFeature as IMCF
from features import ImageSymmetryFeature
from scipy import misc

class D():
    domain = 'image'

    def __init__(self, img):
        self.obj = img

def resave_folder(folder, cm_name='viridis', shape=(1000, 1000)):
    d = os.listdir(folder)
    for fname in d:
        fpath = os.path.join(folder, fname)
        if fpath[-3:] == 'txt':
            png_name = "{}_{}_{}x{}.png".format(fname[:-4], cm_name, shape[0], shape[1])
            #print(png_name)
            if png_name in d:
                print("Already found {}, skipping.".format(png_name))
            elif fname.startswith('f'):
                print("Resaving {} as {}".format(fpath, shape))
                resave(fpath, cm_name, shape)


def resave(fname, cm_name='viridis', shape=(1000,1000)):
    pset = util.create_pset()
    x = np.linspace(0.0, 1.0, 256)
    color_map = None
    if cm_name is not None:
        color_map = cm.get_cmap(cm_name)(x)[np.newaxis, :, :3][0]
    GIA.resave_with_resolution(fname, pset, color_map, shape=shape, cm_name=cm_name)


if __name__ == "__main__":
    folder = "/Users/pihatonttu/git/mas_learning_f17/experiments/collab/examples"
    resave_folder(folder, cm_name=None)
    #fname = "/Users/pihatonttu/uni/own_publications/evoMUSART18/gallery/f_art00058tcp_localhost_5561_1_benford_v0.68_n0.5-tcp_localhost_5563_2_benford.txt"
    #resave(fname, cm_name=None)

#fname = "experiments/collab/collab_test/tcp_localhost_5563_1/f_artifact00002_0.5622450674095718.txt"
#fname = '../gallery/f_art00008_benford-global_contrast_factor_0.849.txt'
#resave(fname)

#fname = 'experiments/gp_test_be200/tcp_localhost_5560_1/artifact00000_0.23518174583937348.png'
#fname = 'experiments/gp_test_be2002/tcp_localhost_5560_1/artifact00016_0.8894606783149983.png'
#fname = 'experiments/gp_test_be2002/tcp_localhost_5560_1/artifact00011_0.8983110672305468.png'
#img = misc.imread(fname)
#a = D(img)
#gca = IGCFF()
#gca(a)
#mc = IMCF()
#ret = mc(a)
#print(ret)

'''
fname = '../scrap/stest.png'
img = misc.imread(fname)
a = D(img)
isf = ImageSymmetryFeature(1, use_entropy=False)
print(isf(a))
isf = ImageSymmetryFeature(2, use_entropy=False)
print(isf(a))
isf = ImageSymmetryFeature(4, use_entropy=False)
print(isf(a))
isf = ImageSymmetryFeature(7, use_entropy=False)
print(isf(a))
'''
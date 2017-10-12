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

def resave(fname):
    pset = util.create_pset()
    cm_name = 'viridis'
    x = np.linspace(0.0, 1.0, 256)
    color_map = cm.get_cmap(cm_name)(x)[np.newaxis, :, :3][0]
    GIA.resave_with_resolution(fname, pset, color_map, shape=(2000, 2000))


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
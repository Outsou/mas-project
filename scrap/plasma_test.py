import numpy as np
from scipy import misc
import utils.primitives

h = 100
w = 100
img = np.zeros((h, w))
t = 0
coords = [(x, y) for x in range(w) for y in range(h)]
for x, y in coords:
    # Normalize coordinates in range [-1, 1]
    x_normalized = x / w * 2 - 1
    y_normalized = y / h * 2 - 1
    img[x, y] = utils.primitives.plasma(x, y, t, 0.005)
misc.imsave('test2.png', img)

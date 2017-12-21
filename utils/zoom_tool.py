import sys, pygame
import numpy as np
from utils import util
from deap import gp
import argparse
from artifacts import GeneticImageArtifact as GIA


if __name__ == '__main__':
    desc = "Zoom your favourite image :)."
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('-s', metavar='s', type=str, dest='s',
                        help="The function in string form.", default=None)
    parser.add_argument('-f', metavar='f', type=str, dest='f',
                        help="Function file.", default=None)
    parser.add_argument('-x', metavar='x', type=int, dest='x',
                        help="Width.", default=200)
    parser.add_argument('-y', metavar='y', type=int, dest='y',
                        help="Height.", default=200)
    args = parser.parse_args()

    shape = (args.x, args.y)

    if args.s is None and args.f is None:
        exit()

    pset = util.create_pset()

    if args.s is not None:
        individual = gp.PrimitiveTree.from_string(args.s, pset)
    else:
        individual = GIA.artifact_from_file(args.f, pset)
    func = gp.compile(individual, pset)

    width, height = shape[0], shape[1]
    coords = [(x, y) for x in range(shape[0]) for y in range(shape[1])]
    zoom_speed = 0.2
    mov_speed = 0.2

    def make_img(rect):
        img = np.zeros(shape)

        left_x = rect[1] / (width - 1) * 2 - 1
        right_x = (rect[1] + rect[3]) / (width - 1) * 2 - 1
        up_y = rect[0] / (height - 1) * 2 - 1
        down_y = (rect[0] + rect[2]) / (height - 1) * 2 - 1

        x_dist = abs(left_x - right_x)
        y_dist = abs(up_y - down_y)

        for x, y in coords:
            x_normalized = x / (width - 1) * x_dist + left_x
            y_normalized = y / (height - 1) * y_dist + up_y
            img[y, x] = func(x_normalized, y_normalized)

        img = np.clip(img, 0, 255, out=img)
        img = np.dstack((img, img, img))
        return np.uint(img)

    center = np.array([(width - 1) / 2, (height - 1) / 2])
    size = [(width - 1), (height - 1)]
    rect = np.array([center[0] - size[0] / 2, center[1] - size[1] / 2, size[0], size[1]])
    img = make_img(rect)
    screen = pygame.display.set_mode(shape)

    while 1:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: sys.exit()
            if event.type == pygame.KEYDOWN:
                old_rect = np.array(rect)
                if event.key == pygame.K_KP_PLUS:
                    size[0] -= size[0] * zoom_speed
                    size[1] -= size[1] * zoom_speed
                if event.key == pygame.K_KP_MINUS:
                    size[0] += size[0] * zoom_speed
                    size[1] += size[1] * zoom_speed
                if event.key == pygame.K_LEFT:
                    center[0] -= size[0] * mov_speed
                if event.key == pygame.K_RIGHT:
                    center[0] += size[0] * mov_speed
                if event.key == pygame.K_UP:
                    center[1] -= size[1] * mov_speed
                if event.key == pygame.K_DOWN:
                    center[1] += size[1] * mov_speed
                if event.key == pygame.K_r:
                    center = np.array([(width - 1) / 2, (height - 1) / 2])
                    size = [(width - 1), (height - 1)]

                size[0] = np.clip(size[0], 0, width)
                size[1] = np.clip(size[1], 0, height)
                center[0] = np.clip(center[0], 0 + size[0] / 2, width - size[0] / 2)
                center[1] = np.clip(center[1], 0 + size[0] / 2, height - size[1] / 2)

                rect = np.array([center[0] - size[0] / 2, center[1] - size[1] / 2, size[0], size[1]])

                if not np.array_equal(old_rect, rect):
                    img = make_img(rect)


        screen.fill((0, 0, 0))
        pygame.surfarray.blit_array(screen, img)
        pygame.display.flip()


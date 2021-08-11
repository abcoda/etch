# True := White

import math
from functools import reduce
import matplotlib.pyplot as plt
from shapely import geometry
from sys import path
from PIL import Image
import numpy as np
import copy
from statistics import mean
import random
from collections import deque
import alphashape
from descartes import PolygonPatch
import pandas as pd

from numpy.core.arrayprint import DatetimeFormat
random.seed(0)


class Node():
    def __init__(self, loc, parent, action):
        self.loc = loc
        self.parent = parent
        self.action = action


class Drawing:
    def __init__(self, source):
        h_orig, w_orig = source.shape
        self.flag_index = 0
        self.h = h_orig * 2
        self.w = w_orig * 2

        # im = Image.fromarray(source)
        # im.show()

        self.data = np.ones((h_orig*2, w_orig*2), dtype='bool')
        for j in range(h_orig):
            for i in range(w_orig):
                self.data[j*2][i*2] = source[j][i]

        # r = 3
        # for j in range(len(self.data)):
        #     for i in range(len(self.data[0])):
        #         neighbor_found = False
        #         for jj in range(-r, r+1, 1):
        #             for ii in range(-r, r+1, 1):
        #                 if i + ii < len(self.data[0]) and i + ii > 0:
        #                     if j + jj < len(self.data) and j + jj > 0:
        #                         if jj or ii:
        #                             if not self.data[j + jj][i + ii]:
        #                                 neighbor_found = True
        #                                 break
        #             if neighbor_found:
        #                 break
        #         if not neighbor_found:
        #             self.data[j][i] = True
        # im = Image.fromarray(self.data)
        # im.show()

        self.blocked = np.zeros((self.h, self.w), dtype='bool')
        self.flagged = np.zeros((self.h, self.w), dtype='bool')

        self.head = (0, 0)

        # manually clear the top left pixel and two pixels around it
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i or j:
                    x = self.head[0] + i
                    y = self.head[1] + j
                    if x >= 0 and y >= 0 and x < self.w and y < self.h:
                        self.data[y][x] = True
        # self.data[0][:2] = True
        # self.data[1][:2] = True

        self.blocked[self.head[1]][self.head[0]] = True

    def is_valid_loc(self, loc):
        x, y = loc
        return x >= 0 and y >= 0 and x < self.w and y < self.h

    def get_neighbors(self, loc):
        x, y = loc
        neighbors = [
            [(x, y - 1), (0, -1)],  # up
            [(x + 1, y), (1, 0)],  # right
            [(x, y + 1), (0, 1)],  # down
            [(x - 1, y), (-1, 0)]  # left
        ]

        valid = [
            neighbor for neighbor in neighbors if self.is_valid_loc(neighbor[0])]
        allowed = [[(x, y), action]
                   for (x, y), action in valid if not self.blocked[y][x]]
        return allowed

    def step(self):
        start = tuple(self.head)
        # connection_distance = 0
        # connect = False
        frontier = deque([Node(start, None, None)])
        explored = set()

        d_max = 5
        d = 1
        while True:
            if len(frontier) == 0:
                return False

            node = frontier.popleft()
            x, y = node.loc

            # if pixel is black, connection point has been found
            if not self.data[y][x] and not self.blocked[y][x]:
                options = [node]
                for item in frontier:
                    if not self.data[item.loc[1]][item.loc[0]] and not self.blocked[item.loc[1]][item.loc[0]]:
                        options.append(item)
                choice = False
                for option in options:
                    min_flag = self.h*self.w + 1
                    if self.flagged[option.loc[1]][option.loc[0]]:
                        if self.flagged[option.loc[1]][option.loc[0]] < min_flag:
                            choice = option
                for option in options:
                    if not self.flagged[option.loc[1]][option.loc[0]]:
                        self.flagged[option.loc[1]
                                     ][option.loc[0]] = self.flag_index
                        self.flag_index += 1
                if not choice:
                    # choice = random.choice(options)
                    choice = options[0]
                node = choice
                actions = []
                cells = []
                while node.parent is not None:
                    actions.append(node.action)
                    cells.append(node.loc)
                    node = node.parent
                actions.reverse()
                cells.reverse()

                for i in range(len(actions)):
                    action = actions[i]
                    cell = cells[i]
                    old_x, old_y = self.head
                    self.blocked[old_y][old_x] = True
                    self.data[old_y][old_x] = False

                    if self.is_valid_loc((old_x, old_y - 1)):
                        self.blocked[old_y - 1][old_x] = True
                    if self.is_valid_loc((old_x, old_y + 1)):
                        self.blocked[old_y + 1][old_x] = True
                    if self.is_valid_loc((old_x + 1, old_y)):
                        self.blocked[old_y][old_x + 1] = True
                    if self.is_valid_loc((old_x - 1, old_y)):
                        self.blocked[old_y][old_x - 1] = True

                    self.head = tuple(cell)
                return True
            explored.add(node.loc)
            for loc, action in self.get_neighbors(node.loc):
                if not any(node.loc == loc for node in frontier) and loc not in explored:
                    frontier.append(Node(loc, node, action))

            d += 1
            # if d > d_max:
            # return False

    def show(self):
        im = Image.fromarray(self.data)
        im.show()

    def save(self, name):
        im = Image.fromarray(self.data)
        im.save(name, format='png')


im = Image.open('contrast.jpg')
# im = Image.open('sliver.jpg')

dithered = im.convert('1')
# dithered.show()

source = np.array(dithered)
drawing = Drawing(source)
data = drawing.data
# print('here')
points = []

for j in range(len(data)):
    for i in range(len(data[0])):
        if not data[j][i]:
            points.append((i, j))

# HULLS METHOD
# hulls = []
# counter = 0
# while len(points) > 2:
#     print(len(points))
#     poly = geometry.Polygon(points)
#     hull = poly.convex_hull
#     x, y = hull.exterior.xy
#     plt.plot(x, y)
#     for xx, yy in zip(x[:-1], y[:-1]):
#         # print((xx, yy))
#         point = (int(xx), int(yy))
#         points.remove(point)
#     hulls.append(hull)
#     counter += 1

# ALPHA SHAPE METHOD
# points_2d = [
#     [1, 1], [1, 2], [2, 1], [2, 2]
# ]
# points_2d = [(0., 0.), (0., 1.), (1., 1.), (1., 0.),
#  (0.5, 0.25), (0.5, 0.75), (0.25, 0.5), (0.75, 0.5)]
# print('here')
# alpha_shape = alphashape.alphashape(points)
# print(alpha_shape)
# fig, ax = plt.subplots()
# ax.scatter(*zip(*points))
# ax.add_patch(PolygonPatch(alpha_shape, alpha=0.2))

# ANGLE METHOD
p = points
center = reduce(lambda a, b: (a[0] + b[0], a[1] + b[1]), p, (0, 0))
center = (center[0] / len(p), (center[1] / len(p)))
p.sort(key=lambda a: math.atan2(a[1] - center[1], a[0] - center[0]))

poly = geometry.Polygon(p)
# hull = poly.convex_hull
x, y = poly.exterior.xy
plt.plot(x, y)
plt.show()

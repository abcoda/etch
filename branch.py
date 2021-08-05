# True := White

from sys import path
from PIL import Image
import numpy as np
import copy
from statistics import mean
import random
from collections import deque

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

        self.data = np.ones((h_orig*2, w_orig*2), dtype='bool')
        for j in range(h_orig):
            for i in range(w_orig):
                self.data[j*2][i*2] = source[j][i]

        self.blocked = np.zeros((self.h, self.w), dtype='bool')
        # self.num_neighbors = np.zeros((self.h, self.w), dtype='int')
        # self.num_connections = np.zeros((self.h, self.w), dtype='int')

        # for j in range(self.h):
        #     for i in range(self.w):
        #         if j > 0:
        #             self.num_neighbors[j][i] += 1
        #         if j < self.h - 1:
        #             self.num_neighbors[j][i] += 1
        #         if i > 0:
        #             self.num_neighbors[j][i] += 1
        #         if i < self.w - 1:
        #             self.num_neighbors[j][i] += 1

        # print(self.num_neighbors)

        # starting point
        self.head = (0, 0)

        # manually set the starting point to black
        # self.data[self.head[1]][self.head[0]] = False

        # manually clear the pixels surrounding the starting point
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i or j:
                    x = self.head[0] + i
                    y = self.head[1] + j
                    if x >= 0 and y >= 0 and x < self.w and y < self.h:
                        self.data[y][x] = True

        # self.blocked[self.head[1]][self.head[0]] = True

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

    def step(self, start):
        start = start
        self.data[start[1]][start[0]] = True
        frontier = deque([Node(start, None, None)])
        explored = set()

        while True:
            if len(frontier) == 0:
                self.data[start[1]][start[0]] = False
                self.blocked[start[1]][start[0]] = True
                return False

            node = frontier.popleft()
            x, y = node.loc

            # if pixel is black and not blocked (one of the original filled in pixels), connection point has been found
            if not self.data[y][x] and not self.blocked[y][x]:
                finish = [x, y]
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
                    if i > 0:
                        self.blocked[old_y][old_x] = True
                        if self.is_valid_loc((old_x, old_y - 1)):
                            self.blocked[old_y - 1][old_x] = True
                        if self.is_valid_loc((old_x, old_y + 1)):
                            self.blocked[old_y + 1][old_x] = True
                        if self.is_valid_loc((old_x + 1, old_y)):
                            self.blocked[old_y][old_x + 1] = True
                        if self.is_valid_loc((old_x - 1, old_y)):
                            self.blocked[old_y][old_x - 1] = True
                    self.data[old_y][old_x] = False
                    self.head = tuple(cell)

                self.blocked[start[1]][start[0]] = False
                self.blocked[finish[1]][finish[0]] = False
                self.head = start
                self.data[self.head[1]][self.head[0]] = True
                return True
            explored.add(node.loc)
            for loc, action in self.get_neighbors(node.loc):
                if not any(node.loc == loc for node in frontier) and loc not in explored:
                    frontier.append(Node(loc, node, action))

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

# drawing.step()
# drawing.step()
# print(drawing.blocked)

options = []
for y in range(drawing.h):
    for x in range(drawing.w):
        if not drawing.data[y][x]:
            options.append((x, y))

# counter = 0
while True:
    if not drawing.step(drawing.head):
        remaining = len(options)
        print(remaining)
        if remaining > 0:
            drawing.head = random.choice(options)
            options.remove(drawing.head)
        else:
            break

        # choice = False
        # for y in range(drawing.h):
            # for x in range(drawing.w):
            # if not drawing.blocked[y][x] and not drawing.data[y][x]:
            # choice = (x, y)
            # break
            # if choice:
            # break
        # if choice:
            # drawing.head = choice
        # else:
            # break
    # counter += 1
    # if counter % 500 == 0:
    #     remaining = 0
    #     for y in range(drawing.h):
    #         for x in range(drawing.w):
    #             if not drawing.blocked[y][x] and not drawing.data[y][x]:
    #                 remaining += 1
    #     print(remaining)
        # print()
        # print(len(options))
        # drawing.head = random.choice(options)
        # print(drawing.head)

        # while True:
        #     if not drawing.step():
        #         break

        # options = []
        # for y in range(drawing.h):
        #     for x in range(drawing.w):
        #         if not drawing.blocked[y][x] and not drawing.data[y][x]:
        #             options.append((x, y))
        #     # options = [(x, y) for x, y in drawing.data[row] for row in drawing.data if not drawing.blocked[y]
        #             #    [x] and not drawing.data[y][x]]
        # if len(options) == 0:
        #     break
        # else:
        #     print(len(options))

        # drawing.head = random.choice(options)
        # for i in range(-1, 2):
        # for j in range(-1, 2):
        # if i or j:
        # x = drawing.head[0] + i
        # y = drawing.head[1] + j
        # if x >= 0 and y >= 0 and x < drawing.w and y < drawing.h:
        # drawing.data[y][x] = True
        # drawing.data[0][:2] = True
        # drawing.data[1][:2] = True

        # drawing.blocked[drawing.head[1]][drawing.head[0]] = True
# print(drawing.blocked)
drawing.show()

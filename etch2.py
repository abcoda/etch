# True := White

from PIL import Image
import numpy as np
import copy
from statistics import mean
import random
from collections import deque
random.seed(0)


class Node():
    def __init__(self, loc, parent, action):
        self.loc = loc
        self.parent = parent
        self.action = action


class Drawing:
    def __init__(self, source):
        h_orig, w_orig = source.shape
        self.h = h_orig * 2
        self.w = w_orig * 2
        self.data = np.ones((h_orig*2, w_orig*2), dtype='bool')
        for j in range(h_orig):
            for i in range(w_orig):
                self.data[j*2][i*2] = source[j][i]

        # manually fill in the top left pixel and clear the pixels around it
        self.data[0][0] = False
        self.data[0][1] = True
        self.data[1][:2] = True

        self.head = [0, 0]
        self.blocked[0][0] = True

        self.blocked = np.zeros((self.h, self.w), dtype='bool')
        self.flagged = np.zeros((self.h, self.w), dtype='bool')

    def get_neighbors(loc, d=1, manhattan=True):
        x, y = loc
        neighbors = np.empty((d*4, 2))
        valid_count = 0
        for i in range(d):
            j = d - i

            neighbors[i] = [x + i, y - j]
            neighbors[i + d] = [x + j, y + i]
            neighbors[i + 2*d] = [x - i, y + j]
            neighbors[i + 3*d] = [x - j, y - i]
        return neighbors

    def is_valid_loc(self, loc):
        x, y = loc
        return x >= 0 and y >= 0 and x < self.w and y < self.h

    def get_closest_dot(self, loc):
        d = 0
        options = []
        selection = False
        while d < 20:
            neighbors = self.get_neighbors(loc, d)
            for x, y in neighbors:
                if self.is_valid_loc((x, y)):
                    if not self.data[y][x] and not self.blocked[y][x]:
                        if self.flagged[y][x] and not selection:
                            selection = [x, y]
                        self.flagged[y][x] = True
                        options.append([x, y])
            if len(options) > 0:
                if selection:
                    return selection
                else:
                    return random.choice(options)
            d += 1
        return False

    def connect(self, a, b):
        return True

    def step(self):
        a = self.head
        b = self.get_closest_dot(a)
        return self.connect(a, b)

    def show(self):
        im = Image.fromarray(self.data)
        im.show()

    def save(self, name):
        im = Image.fromarray(self.data)
        im.save(name, format='png')


im = Image.open('test.jpg')
dithered = im.convert('1')
# dithered.show()

source = np.array(dithered)
drawing = Drawing(source)
drawing.show()

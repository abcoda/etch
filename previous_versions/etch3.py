from tkinter import Tk, Canvas, Frame, BOTH
# True := White

import sys
from sys import path
from PIL import Image
import numpy as np
import copy
from statistics import mean
import random
from collections import deque
import time
import keyboard


# from numpy.core.arrayprint import DatetimeFormat
random.seed(0)


class Display(Frame):

    def __init__(self, root, scale):
        super().__init__()

        self.root = root
        self.initUI()
        self.scale = scale

    def initUI(self):
        self.master.title("Etch")
        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self)
        self.canvas.pack(fill=BOTH, expand=1)

    def load(self, data):
        self.root.geometry(
            f"{len(data[0]) * self.scale}x{len(data) * self.scale}+50+50")

        for row in range(len(data)):
            for col in range(len(data[0])):
                if not data[row][col]:
                    self.canvas.create_rectangle(
                        col * self.scale, row * self.scale, col * self.scale + self.scale, row * self.scale + self.scale, fill='black', width=0)
                elif col % 2 == row % 2:
                    self.canvas.create_rectangle(
                        col * self.scale, row * self.scale, col * self.scale + self.scale, row * self.scale + self.scale, fill='#ddd', width=0)

    def add(self, pixel):
        self.canvas.create_rectangle(pixel[0] * self.scale, pixel[1] * self.scale, pixel[0]
                                     * self.scale + self.scale, pixel[1] * self.scale + self.scale, fill='black', width=0)

    def refresh(self):
        self.root.update()


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
        self.flagged = np.zeros((self.h, self.w), dtype='int')

        self.head = (0, 0)

        # manually clear the top left pixel and two pixels around it
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i or j:
                    x = self.head[0] + i
                    y = self.head[1] + j
                    if x >= 0 and y >= 0 and x < self.w and y < self.h:
                        self.data[y][x] = True

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
        frontier = deque([Node(start, None, None)])
        explored = set()

        d = 1
        while True:
            if len(frontier) == 0:
                return False

            node = frontier.popleft()
            x, y = node.loc

            # if pixel is black, connection point has been found
            if not self.data[y][x] and not self.blocked[y][x]:
                options = [node]
                # check frontier for any other equally distant black pixels
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
                return cells
            explored.add(node.loc)
            for loc, action in self.get_neighbors(node.loc):
                if not any(node.loc == loc for node in frontier) and loc not in explored:
                    frontier.append(Node(loc, node, action))

            d += 1

    def show(self):
        im = Image.fromarray(self.data)
        im.show()

    def save(self, name):
        im = Image.fromarray(self.data)
        im.save(name, format='png')


def main():
    im = Image.open('sliver.jpg')
    # im = Image.open('contrast_small.jpg')
    # im = Image.open('contrast.jpg')
    # im = Image.open('sargent.jpg')

    dithered = im.convert('1')
    # dithered.show()

    source = np.array(dithered)
    drawing = Drawing(source)
    # drawing.head = (int(drawing.w/2), int(drawing.h/2))
    data = drawing.data
    root = Tk()
    scale = 10
    display = Display(root, scale)
    display.load(data)
    root.update()
    # drawing.show()
    while True:
        new_pixels = drawing.step()
        if not new_pixels:
            print('no connection')
            options = []
            for y in range(drawing.h):
                for x in range(drawing.w):
                    if not drawing.blocked[y][x] and not drawing.data[y][x]:
                        options.append((x, y))
            if len(options) == 0:
                break
            else:
                print(len(options))
            drawing.head = random.choice(options)
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i or j:
                        x = drawing.head[0] + i
                        y = drawing.head[1] + j
                        if x >= 0 and y >= 0 and x < drawing.w and y < drawing.h:
                            drawing.data[y][x] = True

            drawing.blocked[drawing.head[1]][drawing.head[0]] = True
        else:

            # print('here')
            keyboard.wait('space')
            for pixel in new_pixels:
                display.add(pixel)
            display.refresh()
    # drawing.show()
    root.mainloop()


if __name__ == '__main__':
    main()

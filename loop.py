from tkinter import Tk, Canvas, Frame, BOTH
from PIL import Image
import numpy as np
from statistics import mean
import random
from collections import deque
import keyboard
import time

COUNTER = 0

# from numpy.core.arrayprint import DatetimeFormat
random.seed(0)


class Display(Frame):

    def __init__(self, scale, root):
        super().__init__()

        self.root = root
        self.initUI()
        self.scale = scale
        self.head = False

    def initUI(self):
        self.master.title("Loop")
        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self)
        self.canvas.pack(fill=BOTH, expand=1)

    def load(self, drawing):
        # anchors = drawing.anchors
        data = drawing.data
        loop = drawing.loop
        self.root.geometry(
            # f"{len(anchors[0]) * self.scale}x{len(anchors) * self.scale}+50+50")
            f"{len(data[0]) * self.scale}x{len(data) * self.scale}+0+0")

        for row in range(len(data)):
            for col in range(len(data[0])):
                if data[row][col]["anchor"]:
                    self.canvas.create_rectangle(
                        col * self.scale, row * self.scale, col * self.scale + self.scale, row * self.scale + self.scale, fill='red', width=0)
                elif col % 2 == row % 2:
                    self.canvas.create_rectangle(
                        col * self.scale, row * self.scale, col * self.scale + self.scale, row * self.scale + self.scale, fill='#ddd', width=0)

    def add(self, pixel, color="black"):
        self.canvas.create_rectangle(pixel[0] * self.scale, pixel[1] * self.scale, pixel[0]
                                     * self.scale + self.scale, pixel[1] * self.scale + self.scale, fill='black', width=0)

    def refresh(self, drawing):
        for row in range(len(drawing.data)):
            for col in range(len(drawing.data[0])):
                if drawing.data[row][col]["anchor"]:
                    self.canvas.create_rectangle(
                        col * self.scale, row * self.scale, col * self.scale + self.scale, row * self.scale + self.scale, fill='red', width=0)
                elif col % 2 == row % 2:
                    self.canvas.create_rectangle(
                        col * self.scale, row * self.scale, col * self.scale + self.scale, row * self.scale + self.scale, fill='#ddd', width=0)
                else:
                    self.canvas.create_rectangle(
                        col * self.scale, row * self.scale, col * self.scale + self.scale, row * self.scale + self.scale, fill='#fff', width=0)

        for pixel in drawing.loop:
            x, y = pixel.loc
            self.canvas.create_rectangle(
                x * self.scale, y * self.scale, x * self.scale + self.scale, y * self.scale + self.scale, fill=('red' if drawing.data[y][x]["anchor"] else 'black'), width=0)

        if self.head:
            x, y = self.head
            self.canvas.create_rectangle(
                x * self.scale, y * self.scale, x * self.scale + self.scale, y * self.scale + self.scale, fill='cyan', width=0)

        # for row in range(len(drawing.data)):
        #     for col in range(len(drawing.data[0])):
        #         if drawing.data[row][col]["inside"] == True:
        #             self.canvas.create_rectangle(
        #                 col * self.scale, row * self.scale, col * self.scale + self.scale, row * self.scale + self.scale, fill=('blue'), width=0)
        #         if drawing.data[row][col]["inside"] == False:
        #             self.canvas.create_rectangle(
        #                 col * self.scale, row * self.scale, col * self.scale + self.scale, row * self.scale + self.scale, fill=('green'), width=0)

        self.root.update()


class Node():
    def __init__(self, loc, parent, action):
        self.loc = loc
        self.parent = parent
        self.action = action


class Pixel:
    def __init__(self, loc, past, future):
        self.loc = loc
        # self.x = loc[0]
        # self.y = loc[1]
        self.past = past
        self.future = future
        # self.anchored = False

    @property
    def x(self):
        return self.loc[0]

    @property
    def y(self):
        return self.loc[1]

    def __str__(self):
        return f"({self.x}, {self.y})"

    # def anchor(self):
        # self.anchored = True


class Drawing:
    def __init__(self, source, display=None):
        h_orig, w_orig = source.shape
        self.flag_index = 0
        self.h = h_orig * 2
        self.w = w_orig * 2
        self.counter = 0
        self.display = display

        self.data = []
        for j in range(self.h):
            self.data.append([])
            for i in range(self.w):
                self.data[j].append({
                    "anchor": False,
                    "inside": None,
                    "occupied": False
                })

        # self.anchors = np.zeros((h_orig*2, w_orig*2), dtype='bool')
        for j in range(h_orig):
            for i in range(w_orig):
                # self.anchors[j*2][i*2] = not source[j][i]
                self.data[j*2][i*2]["anchor"] = not source[j][i]

        # create initial loop along outer boundary (excluding bottom row and right-most column, since they can't have anchor points)
        self.loop = []
        first = Pixel((0, 0), None, None)
        past = first
        self.loop.append(past)
        for i in range(1, self.w - 1):
            current = Pixel((i, 0), past, None)
            self.loop.append(current)
            current.past.future = current
            past = current
        for j in range(1, self.h - 1):
            current = Pixel((self.w - 2, j), past, None)
            self.loop.append(current)
            current.past.future = current
            past = current
        for i in range(self.w - 3, -1, -1):
            current = Pixel((i, self.h - 2), past, None)
            self.loop.append(current)
            current.past.future = current
            past = current
        for j in range(self.h-3, 0, -1):
            current = Pixel((0, j), past, None)
            self.loop.append(current)
            current.past.future = current
            past = current
        current.future = first
        first.past = current

        # anchor all pixels in the loop that rest on an anchor point
        for pixel in self.loop:
            # if self.data[pixel.y][pixel.x]["anchor"]:
            #     pixel.anchor()
            self.data[pixel.y][pixel.x]["occupied"] = True

        # mark all pixels contained inside the loop as "inside", and outside as "not inside". Loop pixels have no value for "inside"
        for j in range(1, self.h - 2):
            for i in range(1, self.w - 2):
                self.data[j][i]["inside"] = True
        # mark right-most column as outside
        for j in range(self.h):
            self.data[j][self.w - 1]["inside"] = False
        # mark bottom row as outside
        for i in range(self.w):
            self.data[self.h-1][i]["inside"] = False

    def get_direction(self, head, vertical, vector=False):
        result = None
        if vertical:
            if self.is_valid_loc((head.x + 1, head.y)):
                direction = self.data[head.y][head.x + 1]["inside"]
                if direction is None:
                    if self.is_valid_loc((head.x - 1, head.y)):
                        result = not self.data[head.y][head.x - 1]["inside"]
                    else:
                        result = True
                else:
                    result = direction
            else:
                result = False
        else:
            if self.is_valid_loc((head.x, head.y - 1)):
                direction = self.data[head.y - 1][head.x]["inside"]
                if direction is None:
                    if self.is_valid_loc((head.x, head.y + 1)):
                        result = not self.data[head.y + 1][head.x]["inside"]
                    else:
                        result = True
                else:
                    result = direction
            else:
                result = False
        if vector:
            if vertical:
                if result:
                    return (1, 0)
                else:
                    return (-1, 0)
            else:
                if result:
                    return (0, -1)
                else:
                    return (0, 1)
        else:
            return result

    def trbverse(self, start=0, stop=-1):
        """
        OPTIMIZATIONS:
        if there are no anchors that can be reached by the segment, dont include it (advancing it will add unnecessary length to loop)

        """
        segments = []
        head = self.loop[start]
        # head = first
        # if the first pixel in loop is an anchor, check if it is a convex corner
        if self.data[head.y][head.x]["anchor"]:
            outside_count = 0
            neighbors = self.get_neighbors(head.loc)
            for neighbor in neighbors:
                if self.data[neighbor[1]][neighbor[0]]["inside"] == False:
                    outside_count += 1
            #  if it's a convex corner, advance the head two positions (or one if there is a change in direction)
            if outside_count >= 2 or (outside_count == 1 and len(neighbors) < 4):
                vertical = head.future.x == head.x
                head = head.future
                if (head.future.x == head.x) == vertical:
                    head = head.future

        visited = []
        vertical = head.future.x == head.x

        # Value of True for direction is Right or Up, False is Left or Down
        segment = {"pixels": [head], "vertical": vertical,
                   "direction": self.get_direction(head, vertical)}
        while True:
            # advance the head
            head = head.future

            # passed = True
            # if vertical:
            #     if segment["direction"]:  # right
            #         if not self.data[head.y][head.x + 2]["inside"]:
            #             passed = False
            #     else:  # left
            #         if not self.data[head.y][head.x - 2]["inside"]:
            #             passed = False
            # else:
            #     if segment["direction"]:  # up
            #         if not self.data[head.y - 2][head.x]["inside"]:
            #             passed = False
            #     else:  # down
            #         if not self.data[head.y + 2][head.x]["inside"]:
            #             passed = False
            # if not passed:
            #     vertical = head.future.x == head.future.future.x
            #     segment = {"pixels": [head.future], "vertical": vertical,
            #                "direction": self.get_direction(head.future, vertical)}
            #     continue

            # if we've already been to the head, we are done traversing
            if head in visited:
                # SHOULD CHECK FOR CONVEX CORNER HERE
                segments.append(segment)
                break
            visited.append(head)

            # add the new head to the segment
            segment["pixels"].append(head)

            # if the orientation changes after this pixel, we are at the end of a segment (a corner)
            if (head.future.x == head.x) != vertical:
                # if the corner is anchored, check if it is convex
                if self.data[head.y][head.x]["anchor"]:
                    outside_count = 0
                    neighbors = self.get_neighbors(head.loc)
                    for neighbor in neighbors:
                        if self.data[neighbor[1]][neighbor[0]]["inside"] == False:
                            outside_count += 1
                    #  if it is convex, remove it (and the pixel before it if not another corner) from the segment
                    # then advance the head to two pixels after the corner (or one of there is a second corner)
                    if outside_count >= 2 or (outside_count == 1 and len(neighbors) < 4):
                        vertical = head.past.x == head.x
                        if (head.past.past.x == head.past.x) == vertical:
                            del segment["pixels"][-2:]
                        else:
                            del segment["pixels"][-1]

                        vertical = head.future.x == head.x
                        head = head.future
                        if (head.future.x == head.x) == vertical:
                            head = head.future
                # add the completed segment to the list of segments and get the new orientation
                segments.append(segment)
                vertical = head.future.x == head.x
                # create the next segment starting at the end of the last segment
                segment = {"pixels": [head], "vertical": vertical,
                           "direction": self.get_direction(head, vertical)}
            # if there is not a corner but there is an anchor, end the segment and start the next segment at the same position with the same orientation
            elif self.data[head.y][head.x]["anchor"]:
                segments.append(segment)
                segment = {"pixels": [head], "vertical": vertical,
                           "direction": self.get_direction(head, vertical)}
        return segments

    def traverse(self, start=0, stop=-1):
        """
        OPTIMIZATIONS:
        if there are no anchors that can be reached by the segment, dont include it (advancing it will add unnecessary length to loop)

        """

        segments = []
        head = self.loop[start]
        visited = []
        pixels = []
        vertical = head.future.x == head.x
        dir_vect = self.get_direction(head, vertical, vector=True)
        direction = self.get_direction(head, vertical)
        flag = False
        update = False
        while True:
            # if self.counter > 26 and head.loc == (5, 24):
            #     update = True
            if update:
                self.display.head = head.loc
                self.display.refresh(self)
            if head in visited:
                break
            future = (head.x + dir_vect[0], head.y + dir_vect[1])

            neighbors = self.get_neighbors(future)
            neighbor_count = 0
            for neighbor in neighbors:
                if self.data[neighbor[1]][neighbor[0]]["occupied"]:
                    neighbor_count += 1

            # if the pixel's future is directly against another pixel, don't include it in the segment, end the segment, and start a new one
            if neighbor_count > 1:
                if len(pixels) > 2:
                    segments.append({
                        "pixels": pixels,
                        "vertical": vertical,
                        "direction": direction
                    })
                    # print_segments(segments)

                pixels = []
                if (head.future.x == head.x) == vertical:
                    visited.append(head)
                    head = head.future
                else:
                    vertical = not vertical
                    direction = self.get_direction(head, vertical)
                    dir_vect = self.get_direction(head, vertical, vector=True)
                    flag = False

            # if the pixel is valid, but it is an anchor or a corner, include it in the segment and then start a new one
            elif self.data[head.y][head.x]['anchor'] or (head.future.x == head.x) != vertical or (head.past.x == head.x) != vertical:
                pixels.append(head)
                if flag:
                    flag = False
                    head = head.future
                else:
                    if len(pixels) > 2:
                        segments.append({
                            "pixels": pixels,
                            "vertical": vertical,
                            "direction": direction
                        })
                    # print_segments(segments)
                    pixels = []
                    flag = True

                # print(segments)
                vertical = head.future.x == head.x
                direction = self.get_direction(head, vertical)
                dir_vect = self.get_direction(head, vertical, vector=True)

            # otherwise, just add the pixel to the current segment, advance the head, and keep going unless the head has been visited
            else:
                pixels.append(head)
                visited.append(head)
                head = head.future
                # if head in visited:
                #     break
        return segments

    def step(self):
        # find the longest segment in the loop
        segments = self.traverse()
        if len(segments) > 0:
            segment = max(segments, key=lambda segment: len(segment["pixels"]))
        else:
            return False
        # OPTIMIZATION: if multiple segmentso of max length, pick the one that is closest to reaching an anchor
        # only consider segments with length of at least 3 (cant advance 1 or 2 pixel segments)
        if len(segment["pixels"]) < 3:
            print("No valid segments")
            return False
        # get a vector representing the direction of the inside of the loop relative to the segment
        # True is up (-y) or right(+x)
        vect = (0, 0)
        if segment["vertical"]:
            if segment["direction"]:
                vect = (1, 0)
            else:
                vect = (-1, 0)
        else:
            if segment["direction"]:
                vect = (0, -1)
            else:
                vect = (0, 1)

        # after advancing the segment, the first pixel of the segment will be in new_first_loc
        first = segment["pixels"][0]
        new_first_loc = (first.loc[0] + vect[0], first.loc[1] + vect[1])

        # if the new first location is already in the loop..
        # (this only happens if advancing an unanchored convex corner, can be optimized by making starting loop eliminate these corners)
        if self.data[new_first_loc[1]][new_first_loc[0]]["occupied"]:
            # print("111111111111")
            new_first = first.past
            first.past.future = first.future
            first.future.past = first.past
            self.data[first.loc[1]][first.loc[0]]["occupied"] = False
            self.data[first.loc[1]][first.loc[0]]["inside"] = False
            self.loop.remove(first)
        else:
            new_first = Pixel(
                (first.loc[0] + vect[0], first.loc[1] + vect[1]), first, first.future)
            first.future.past = new_first
            first.future = new_first
            self.data[new_first_loc[1]][new_first_loc[0]]["inside"] = None
            self.data[new_first_loc[1]][new_first_loc[0]]["occupied"] = True
            self.loop.append(new_first)

        last = segment["pixels"][-1]
        new_last_loc = (last.loc[0] + vect[0], last.loc[1] + vect[1])
        # new_last = None

        #  if the new position for the last pixel is already occupied...
        # (this only happens if advancing an unanchored convex corner, can be optimized by making starting loop eliminate these corners)
        if self.data[new_last_loc[1]][new_last_loc[0]]["occupied"]:
            # print('22222222222')
            new_last = last.future
            last.future.past = last.past
            last.past.future = last.future
            self.data[last.loc[1]][last.loc[0]]["occupied"] = False
            self.data[last.loc[1]][last.loc[0]]["inside"] = False
            self.loop.remove(last)
        else:
            new_last = Pixel(
                (last.loc[0] + vect[0], last.loc[1] + vect[1]), last.past, last)
            last.past.future = new_last
            last.past = new_last
            self.data[new_last_loc[1]][new_last_loc[0]]["inside"] = None
            self.data[new_last_loc[1]][new_last_loc[0]]["occupied"] = True

            self.loop.append(new_last)

        # for all of the pixels in between the first and last, just advance them towards the inside without changing past or future
        for pixel in segment["pixels"][1:-1]:
            self.data[pixel.y][pixel.x]["occupied"] = False
            self.data[pixel.y][pixel.x]["inside"] = False
            pixel.loc = (pixel.loc[0] + vect[0], pixel.loc[1] + vect[1])
            self.data[pixel.y][pixel.x]["occupied"] = True
            self.data[pixel.y][pixel.x]["inside"] = None

        # head = new_first
        # while True:
        #     # if self.data[head.y][head.x]["anchor"]:
        #     #     head.anchor()
        #     head = head.future
        #     if head == new_last:
        #         break
        return True

    def is_valid_loc(self, loc):
        x, y = loc
        return x >= 0 and y >= 0 and x < self.w and y < self.h

    def get_neighbors(self, loc):
        x, y = loc
        # neighbors = [
        #     [(x, y - 1), (0, -1)],  # up
        #     [(x + 1, y), (1, 0)],  # right
        #     [(x, y + 1), (0, 1)],  # down
        #     [(x - 1, y), (-1, 0)]  # left
        # ]
        neighbors = [
            (x, y - 1),  # up
            (x + 1, y),  # right
            (x, y + 1),   # down
            (x - 1, y)  # left
        ]

        valid = [
            neighbor for neighbor in neighbors if self.is_valid_loc(neighbor)]
        return valid

    # def step(self, current, previous):
    #     x, y = current
    #     neighbors = self.get_neighbors(current)

    def show(self):
        im = Image.fromarray(self.data)
        im.show()

    def save(self, name):
        im = Image.fromarray(self.data)
        im.save(name, format='png')


def print_segments(segments):
    # segments = drawing.traverse()
    for segment in segments:
        pixels = segment["pixels"]
        vertical = segment["vertical"]
        direction = segment["direction"]
        print(len(pixels), end=', ')
        if vertical:
            print('Vertical', end=', ')
            if direction:
                print('Right')
            else:
                print('Left')
        else:
            print('Horizontal', end=', ')
            if direction:
                print('Up')
            else:
                print('Down')
    print('\n')
    # longest = max(segments, key=lambda segment: len(segment["pixels"]))
    # print(
    # f"Longest: {longest['pixels'][0].loc} to {longest['pixels'][-1].loc}")


def main():
    # im = Image.open('sliver.jpg')
    # im = Image.open('contrast_small.jpg')
    im = Image.open('contrast_tiny.jpg')
    # im = Image.open('contrast.jpg')
    # im = Image.open('sargent.jpg')

    dithered = im.convert('1')
    source = np.array(dithered)
    drawing = Drawing(source)
    scale = 1
    root = Tk()
    display = Display(scale, root)
    display.load(drawing)
    display.refresh(drawing)
    drawing.display = display
    # display.update()
    while True:
        # print_segments(drawing.traverse())
        # print(drawing.counter)
        # keyboard.wait('space')
        if not drawing.step():
            break
        if drawing.counter > 50:
            drawing.counter = 0
            display.refresh(drawing)

        # display.refresh(drawing)
        drawing.counter += 1
        # print(drawing.counter)

    # segments = drawing.traverse()
    # for segment in segments:
    #     pixels = segment["pixels"]
    #     vertical = segment["vertical"]
    #     direction = segment["direction"]
    #     print(len(pixels), end=', ')
    #     if vertical:
    #         print('Vertical', end=', ')
    #         if direction:
    #             print('Right')
    #         else:
    #             print('Left')
    #     else:
    #         print('Horizontal', end=', ')
    #         if direction:
    #             print('Up')
    #         else:
    #             print('Down')
    # longest = max(segments, key=lambda segment: len(segment["pixels"]))
    # print(
    #     f"Longest: {longest['pixels'][0].loc} to {longest['pixels'][-1].loc}")

    print('done')
    display.refresh(drawing)
    root.mainloop()


if __name__ == '__main__':
    main()

from tkinter import Tk, Canvas, Frame, BOTH
from PIL import Image
import numpy as np
import random
import keyboard

random.seed(0)


class Display(Frame):

    def __init__(self, root, drawing, scale=1):
        super().__init__()

        self.root = root
        self.initUI()
        self.scale = scale
        self.head = False
        self.drawing = drawing
        self.root.geometry(
            f"{self.drawing.w * self.scale}x{self.drawing.h * self.scale}+0+0")

    def initUI(self):
        self.master.title("Loop")
        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self)
        self.canvas.pack(fill=BOTH, expand=1)

    def add(self, loc, color="black", width=0):
        self.canvas.create_rectangle(loc[0] * self.scale, loc[1] * self.scale, loc[0]
                                     * self.scale + self.scale, loc[1] * self.scale + self.scale, fill=color, width=width)

    def refresh(self):
        for row in range(self.drawing.h):
            for col in range(self.drawing.w):
                pixel = self.drawing.data[row][col]
                loc = (col, row)
                if pixel.anchor:
                    self.add(loc, color='red')
                elif pixel.inside == None:
                    self.add(loc, color='black')
                elif pixel.inside:
                    self.add(loc, color='blue')
                elif pixel.inside == False:
                    self.add(loc, color='green')
                elif col % 2 == row % 2:
                    self.add((col, row), color='#ddd')
                else:
                    self.canvas.create_rectangle(
                        col * self.scale, row * self.scale, col * self.scale + self.scale, row * self.scale + self.scale, fill='#fff', width=0)

        if self.head:
            x, y = self.head
            self.add(self.head, color='cyan')

        self.root.update()
        return True


class Pixel:
    def __init__(self, loc, past=None, future=None, inside=None, anchor=None):
        self.loc = loc
        self.past = past
        self.future = future
        self.inside = inside
        self.anchor = anchor

    @property
    def x(self):
        return self.loc[0]

    @property
    def y(self):
        return self.loc[1]

    @property
    def in_loop(self):
        return self.inside == None

    @property
    def outside(self):
        if self.inside == False:
            return True
        elif self.inside == True:
            return False
        return None

    def __str__(self):
        return f"({self.x}, {self.y})"


class Drawing:
    def __init__(self, source, display=None):
        h_orig, w_orig = source.shape
        self.h = h_orig * 2
        self.w = w_orig * 2
        self.counter = 0
        self.display = display

        self.data = []
        for j in range(self.h):
            self.data.append([])
            for i in range(self.w):
                self.data[j].append(Pixel(
                    (i, j),
                    anchor=(not source[int(j/2)][int(i/2)] if (i %
                                                               2 == 0 and j % 2 == 0) else False)
                ))

        # create initial loop along outer boundary (excluding bottom row and right-most column, since they can't have anchor points)
        data = self.data
        first = data[0][0]
        past = first
        for i in range(1, self.w - 1):
            current = data[0][i]
            current.past = past
            current.past.future = current
            past = current
        for j in range(1, self.h - 1):
            current = data[j][self.w - 2]
            current.past = past
            current.past.future = current
            past = current
        for i in range(self.w - 3, -1, -1):
            current = data[self.h - 2][i]
            current.past = past
            current.past.future = current
            past = current
        for j in range(self.h-3, 0, -1):
            current = data[j][0]
            current.past = past
            current.past.future = current
            past = current
        current.future = first
        first.past = current

        # mark all pixels contained inside the loop as "inside", and outside as "not inside". Loop pixels have None value for "inside"
        for j in range(1, self.h - 2):
            for i in range(1, self.w - 2):
                self.data[j][i].inside = True
        # mark right-most column as outside
        for j in range(self.h):
            self.data[j][self.w - 1].inside = False
        # mark bottom row as outside
        for i in range(self.w):
            self.data[self.h-1][i].inside = False

        loop = []
        pixel = self.data[0][0]
        while True:
            loop.append(pixel)
            pixel = pixel.future
            if pixel == loop[0]:
                break
        self.loop = loop

    def get_direction(self, pixel, vertical, vector=False):
        result = None
        if vertical:
            if self.is_valid_loc((pixel.x + 1, pixel.y)):
                direction = self.data[pixel.y][pixel.x + 1].inside
                if direction is None:
                    if self.is_valid_loc((pixel.x - 1, pixel.y)):
                        result = not self.data[pixel.y][pixel.x - 1].inside
                    else:
                        result = True
                else:
                    result = direction
            else:
                result = False
        else:
            if self.is_valid_loc((pixel.x, pixel.y - 1)):
                direction = self.data[pixel.y - 1][pixel.x].inside
                if direction is None:
                    if self.is_valid_loc((pixel.x, pixel.y + 1)):
                        result = not self.data[pixel.y + 1][pixel.x].inside
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

        return True

    def is_valid_loc(self, loc):
        x, y = loc
        return x >= 0 and y >= 0 and x < self.w and y < self.h

    def get_neighbors(self, loc):
        x, y = loc

        neighbors = [
            (x, y - 1),  # up
            (x + 1, y),  # right
            (x, y + 1),   # down
            (x - 1, y)  # left
        ]

        valid = [
            neighbor for neighbor in neighbors if self.is_valid_loc(neighbor)]
        return valid

    def show(self):
        im = Image.fromarray(self.data)
        im.show()

    def save(self, name):
        im = Image.fromarray(self.data)
        im.save(name, format='png')


def print_segments(segments):
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


def main():
    im = Image.open('sliver.jpg')
    # im = Image.open('contrast_small.jpg')
    # im = Image.open('contrast_tiny.jpg')
    # im = Image.open('contrast.jpg')
    # im = Image.open('sargent.jpg')

    dithered = im.convert('1')
    source = np.array(dithered)
    drawing = Drawing(source)
    root = Tk()
    display = Display(root, drawing, scale=10)
    # display.load()
    # display.refresh()
    # drawing.display = display
    # while True:
    #     # keyboard.wait('space')
    #     if not drawing.step():
    #         break
    #     if drawing.counter > 50:
    #         drawing.counter = 0
    #         display.refresh(drawing)
    #     drawing.counter += 1

    display.refresh()
    root.mainloop()


if __name__ == '__main__':
    main()

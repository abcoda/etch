from tkinter import Tk, Canvas, Frame, BOTH
from PIL import Image
import numpy as np
import random
import keyboard
import time

random.seed(0)


class Display(Frame):

    def __init__(self, root, drawing, scale=1, ext=False):
        super().__init__()

        self.root = root
        self.initUI()
        self.scale = scale
        self.head = False
        self.drawing = drawing
        self.ext = ext
        if ext:
            self.root.geometry(
                f"{self.drawing.w*2 * self.scale}x{self.drawing.h*3 * self.scale + 3*scale}+0+0")

        else:
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
        self.canvas.delete('all')
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
                    self.add(loc, color='#ddd')
                else:
                    self.add(loc, color='white')
        if self.ext:
            offset = self.drawing.h + 2
            for pixel in self.drawing.loop:
                loc = pixel.loc
                future_loc = pixel.future.loc
                vect = (future_loc[0]-loc[0], future_loc[1]-loc[1])
                middle_loc = (loc[0] + vect[0]/2, loc[1] + vect[1]/2)
                loc = (loc[0] * 2, loc[1] * 2 + offset)
                middle_loc = (middle_loc[0] * 2, middle_loc[1] * 2 + offset)

                self.add(loc, color='black')
                self.add(middle_loc)
            for row in range(0, self.drawing.h):
                for col in range(0, self.drawing.w):
                    pixel = self.drawing.data[row][col]
                    loc = (col*2, row*2 + offset)
                    if pixel.anchor:
                        self.add(loc, color='red')
                    # elif pixel.inside == None:
                    #     self.add(loc, color='black')
                    # elif pixel.inside:
                    #     self.add(loc, color='blue')
                    # elif pixel.inside == False:
                    #     self.add(loc, color='green')
                    # elif col % 2 == row % 2:
                    #     self.add(loc, color='#ddd')
                    # else:
                    #     self.add(loc, color='white')
        # if self.head:
        #     x, y = self.head
        #     self.add(self.head, color='cyan')

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
        self.counter = 0
        self.h, self.w = source.shape
        self.display = display

        self.data = []
        for j in range(self.h):
            self.data.append([])
            for i in range(self.w):
                self.data[j].append(Pixel(
                    (i, j),
                    anchor=not source[j][i]
                ))

        # create initial loop along outer boundary (excluding bottom row and right-most column, since they can't have anchor points)
        data = self.data
        first = data[0][0]
        past = first
        for i in range(self.w):
            current = data[0][i]
            current.past = past
            current.past.future = current
            past = current
        for j in range(1, self.h):
            current = data[j][self.w - 1]
            current.past = past
            current.past.future = current
            past = current
        for i in range(self.w - 2, -1, -1):
            current = data[self.h - 1][i]
            current.past = past
            current.past.future = current
            past = current
        for j in range(self.h-2, 0, -1):
            current = data[j][0]
            current.past = past
            current.past.future = current
            past = current
        current.future = first
        first.past = current

        # mark all pixels contained inside the loop as "inside", and outside as "not inside". Loop pixels have None value for "inside"
        for j in range(1, self.h - 1):
            for i in range(1, self.w - 1):
                self.data[j][i].inside = True

        loop = []
        pixel = self.data[0][0]
        while True:
            loop.append(pixel)
            pixel = pixel.future
            if pixel == loop[0]:
                break
        self.loop = loop

    def get_direction(self, pixel, vertical, vector=False):
        x, y = pixel.loc
        # handle convex corners?
        if vertical:
            try:
                if self.data[y][x + 1].inside:
                    if vector:
                        return (1, 0)
                    return True
            except:
                pass
            try:
                if self.data[y][x-1].inside:
                    if vector:
                        return (-1, 0)
                    return False
            except:
                pass
        else:
            try:
                if self.data[y-1][x].inside:
                    if vector:
                        return (0, -1)
                    return True
            except:
                pass
            try:
                if self.data[y+1][x].inside:
                    if vector:
                        return (0, 1)
                    return False
            except:
                pass
        return None

    def traverse(self, start=0, stop=None):
        """
        OPTIMIZATIONS:
        if there are no anchors that can be reached by the segment, dont include it (advancing it will add unnecessary length to loop)

        """
        segments = []
        head = self.loop[start]
        stop = head.future
        anchor_flag = False
        stop_flag = False
        while True:
            if head is stop:
                if stop_flag:
                    break
                stop_flag = True
            vertical = head.future.x == head.x
            direction = self.get_direction(head, vertical, vector=True)
            pixels = []
            while True:
                if self.get_direction(head, vertical) is not None:
                    pixels.append(head)
                    if head.anchor or (head.future.x == head.x) != vertical:
                        if anchor_flag:
                            # head = head.future
                            anchor_flag = False
                        else:
                            anchor_flag = True
                            break
                    else:
                        anchor_flag = False
                    head = head.future

                else:
                    head = head.future
                    break
            if len(pixels) > 1:
                segments.append({
                    "pixels": pixels,
                    "vertical": vertical,
                    "direction": direction
                })
        return segments

    def step(self):
        # find the longest segment in the loop
        segments = self.traverse()
        # print_segments(segments)
        try:
            segment = max(segments, key=lambda segment: len(segment["pixels"]))
        except:
            print("No segments")
            return False
        # OPTIMIZATION: if multiple segmentso of max length, pick the one that is closest to reaching an anchor
        # only consider segments with length of at least 3 (cant advance 1 or 2 pixel segments)

        vect = segment["direction"]

        # after advancing the segment, the first pixel of the segment will be in new_first_loc
        first = segment["pixels"][0]
        new_first = self.data[first.loc[1] + vect[1]][first.loc[0] + vect[0]]
        new_first.past = first
        first.future = new_first
        self.data[new_first.y][new_first.x].inside = None
        self.loop.append(new_first)

        # for all of the pixels in between the first and last, just advance them towards the inside without changing past or future
        past = new_first
        for pixel in segment["pixels"][1:-1]:
            new_pixel = self.data[pixel.loc[1] +
                                  vect[1]][pixel.loc[0] + vect[0]]
            new_pixel.past = past
            new_pixel.past.future = new_pixel
            new_pixel.inside = None
            pixel.inside = False
            pixel.future = None
            pixel.past = None
            past = new_pixel
            self.loop.remove(pixel)
            self.loop.append(new_pixel)
            # self.data[pixel.y][pixel.x].inside = False
            # pixel.loc = (pixel.loc[0] + vect[0], pixel.loc[1] + vect[1])
            # self.data[pixel.y][pixel.x].inside = None
        last = segment["pixels"][-1]
        new_last = self.data[last.loc[1] + vect[1]][last.loc[0] + vect[0]]
        new_last.past = past
        new_last.past.future = new_last
        new_last.future = last
        # last.past.future = new_last
        last.past = new_last
        self.data[new_last.y][new_last.x].inside = None
        self.loop.append(new_last)

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

        im_data = np.ones((self.h*2 + 1, self.w*2 + 1), dtype='bool')

        for pixel in self.loop:
            loc = pixel.loc
            future_loc = pixel.future.loc
            vect = (future_loc[0]-loc[0], future_loc[1]-loc[1])

            im_data[1+pixel.y*2][1+pixel.x*2] = False
            im_data[1+pixel.y*2 + vect[1]][1+pixel.x*2+vect[0]] = False

        im = Image.fromarray(im_data)
        im.save(name, format='png')


def print_segments(segments):
    for segment in segments:
        pixels = segment["pixels"]
        vertical = segment["vertical"]
        direction = segment["direction"]
        print(len(pixels), end=', ')
        if vertical:
            print('Vertical', end=', ')
            if direction[0] == 1:
                print('Right')
            else:
                print('Left')
        else:
            print('Horizontal', end=', ')
            if direction[1] == -1:
                print('Up')
            else:
                print('Down')
    print('')


def main():
    start_time = time.time()
    # im, scale = Image.open('sliver.jpg'), 10
    # im, scale = Image.open('contrast_small.jpg'), 1
    # im, scale = Image.open('contrast_tiny.jpg'), 1
    im, scale = Image.open('face_tiny.jpg'), 1
    # im, scale = Image.open('contrast_micro.jpg'), 2
    # im = Image.open('contrast.jpg')
    # im = Image.open('sargent.jpg')

    dithered = im.convert('1')
    source = np.array(dithered)
    drawing = Drawing(source)
    root = Tk()
    display = Display(root, drawing, scale=scale, ext=True)
    display.refresh()

    while True:
        # keyboard.wait('space')
        if not drawing.step():
            display.refresh()
            break
        # if drawing.counter > 200:
        #     drawing.counter = 0
        #     display.refresh()
        drawing.counter += 1

    end_time = time.time()
    print("Time: ", int(end_time-start_time))
    drawing.save('test.png')
    root.mainloop()


if __name__ == '__main__':
    main()

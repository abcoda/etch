from tkinter import Tk, Canvas, Frame, BOTH
from PIL import Image
import numpy as np
import keyboard
import time
# import random
# random.seed(0)

"""
Display: Class for a UI Window which displays the drawing in its current state.
Each display object is linked to a drawing.
Mainly for debugging purposes.
"""


class Display(Frame):

    def __init__(self, root, drawing, scale=1, ext=False):
        super().__init__()

        self.root = root
        self.initUI()
        self.scale = scale
        self.drawing = drawing

        # If ext is passed in as True, the display will also show the "extended" form of the drawing,
        # which makes the loop effect visible (this is how the final drawings will be displayed)
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

    # Adds a single filled in pixel to the canvas
    def add(self, loc, color="black", width=0):
        self.canvas.create_rectangle(loc[0] * self.scale, loc[1] * self.scale, loc[0]
                                     * self.scale + self.scale, loc[1] * self.scale + self.scale, fill=color, width=width)

    # Updates the canvas with the current state of the drawing
    def refresh(self):
        self.canvas.delete('all')
        for row in range(self.drawing.h):
            for col in range(self.drawing.w):
                # Fill in pixels with different colors depending on the status of the pixel
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
        # If applicable, draw the extended form of the drawing below the original
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

        self.root.update()
        return True


"""
The Pixel object holds the data for a single pixel in the drawing.
    - loc: The (x,y) coordinates of the pixel
    - past: The Pixel object that precedes the current Pixel in the loop (if the pixel is part of the loop)
    - future: The Pixel that comes after the current Pixel (if the current pixel is part of the loop)
    - inside: Denotes whether the Pixel is 
        True: inside of the loop
        False: outside of the loop
        None: part of the loop
    - anchor (bool): Denotes whether the Pixel is an anchor point of the drawing (one of the original pixels in the image)
    - segments: a list of all line segments that the Pixel is a part of 
        segments start and stop at corners, anchor points, or any position that cannot be advanced
        due to another segment blocking it (see the "step" and "traverse" functions)
"""


class Pixel:
    def __init__(self, loc, past=None, future=None, inside=None, anchor=None):
        self.loc = loc
        self.past = past
        self.future = future
        self.inside = inside
        self.anchor = anchor
        self.segments = []

    @property
    def x(self):
        return self.loc[0]

    @property
    def y(self):
        return self.loc[1]

    def __str__(self):
        return f"({self.x}, {self.y})"


"""Represents the Entire Drawing, including all of the pixel data, loop data, 
        and list of segments in the loop
    The source passed in to the drawing is a numpy array of 1's and 0's 
        coming from a strictly black and white image (see main)
    
"""


class Drawing:
    def __init__(self, source, display=None):
        # counter counts the number of iterations of the step function, for debugging or animation purposes
        self.counter = 0
        self.h, self.w = source.shape
        self.display = display
        self.segments = []

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
        self.segments = self.traverse()

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

    def traverse(self, start=False, stop=False):
        """
        OPTIMIZATIONS:
        if there are no anchors that can be reached by the segment, dont include it (advancing it will add unnecessary length to loop)

        """
        if not start:
            start = self.loop[0]
        head = start
        if start.anchor:
            anchor_flag = True
        else:
            anchor_flag = False
        if not stop:
            stop = start
            full_loop = True
            # stop_flag = False
        else:
            full_loop = False
            stop = stop.future
            # stop_flag = True
        stop_flag = False

        segments = []
        while True:
            # if head is stop:
            if stop_flag:
                break
            # stop_flag = True

            vertical = head.future.x == head.x
            direction = self.get_direction(head, vertical, vector=True)
            pixels = []
            while True:
                if self.get_direction(head, vertical) is not None:
                    pixels.append(head)
                    if head.anchor or (head.future.x == head.x) != vertical:
                        if anchor_flag:
                            anchor_flag = False
                        else:
                            anchor_flag = True
                            break
                    else:
                        anchor_flag = False
                    head = head.future
                    if head is stop:
                        stop_flag = True
                        if not full_loop:
                            break

                else:
                    if head.anchor or (head.future.x == head.x) != vertical:
                        if anchor_flag:
                            anchor_flag = False
                            head = head.future
                            if head is stop:
                                stop_flag = True
                            break
                        else:
                            anchor_flag = True
                            break
                    # if head == stop and stop_flag:
                        # break
                    head = head.future
                    if head is stop:
                        stop_flag = True
                    break

            if len(pixels) > 1:
                segment = {
                    "pixels": pixels,
                    "vertical": vertical,
                    "direction": direction
                }
                segments.append(segment)
                for pixel in pixels:
                    if segment not in pixel.segments:
                        pixel.segments.append(segment)

        return segments

    def step(self):
        # find the longest segment in the loop
        segments = self.segments
        # print_segments(segments)
        try:
            # segment = segments[0]
            segment = max(segments, key=lambda segment: len(segment["pixels"]))
        except:
            print("No segments")
            return False
        # OPTIMIZATION: if multiple segmentso of max length, pick the one that is closest to reaching an anchor
        seg_ind = self.segments.index(segment)
        vect = segment["direction"]
        # after advancing the segment, the first pixel of the segment will be in new_first_loc
        first = segment["pixels"][0]
        new_first = self.data[first.loc[1] + vect[1]][first.loc[0] + vect[0]]
        new_first.past = first
        first.future = new_first
        self.data[new_first.y][new_first.x].inside = None
        self.loop.append(new_first)
        first.segments.remove(segment)
        # for seg in first.segments:
        #     redo_segments.append(seg)

        # for all of the pixels in between the first and last, just advance them towards the inside without changing past or future
        flagged = []
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
            pixel.segments = []
            one_further = self.data[new_pixel.y +
                                    vect[1]][new_pixel.x + vect[0]]
            if one_further.inside == None:
                flagged.append(one_further)

        last = segment["pixels"][-1]
        new_last = self.data[last.loc[1] + vect[1]][last.loc[0] + vect[0]]
        new_last.past = past
        new_last.past.future = new_last
        new_last.future = last
        last.past = new_last
        self.data[new_last.y][new_last.x].inside = None
        self.loop.append(new_last)
        last.segments.remove(segment)
        # for seg in last.segments:
        #     redo_segments.append(seg)

        # new_segment = self.traverse(start=new_first,stop=new_last)
        # self.display.refresh()
        self.segments.remove(segment)

        if segment["vertical"]:
            if segment["direction"][0] == 1:
                side1 = self.data[new_last.y - 1][new_last.x]
                side2 = self.data[new_first.y + 1][new_first.x]
                side3 = self.data[new_last.y][new_last.x + 1]
                side4 = self.data[new_first.y][new_first.x + 1]
            else:
                side1 = self.data[new_first.y - 1][new_first.x]
                side2 = self.data[new_last.y + 1][new_last.x]
                side3 = self.data[new_last.y][new_last.x - 1]
                side4 = self.data[new_first.y][new_first.x - 1]
        else:
            if segment["direction"][1] == 1:
                side1 = self.data[new_first.y][new_first.x - 1]
                side2 = self.data[new_last.y][new_last.x + 1]
                side3 = self.data[new_last.y + 1][new_last.x]
                side4 = self.data[new_first.y + 1][new_first.x]
            else:
                side1 = self.data[new_last.y][new_last.x - 1]
                side2 = self.data[new_first.y][new_first.x + 1]
                side3 = self.data[new_first.y - 1][new_first.x]
                side4 = self.data[new_last.y - 1][new_last.x]
        if side1.inside == None:
            flagged.append(side1)
        if side2.inside == None:
            flagged.append(side2)
        if side3.inside == None:
            flagged.append(side3)
        if side4.inside == None:
            flagged.append(side4)

        redo_segments = []
        for pixel in flagged:
            for seg in pixel.segments:
                if seg not in redo_segments:
                    redo_segments.append(seg)
        for seg in redo_segments:
            for pixel in seg["pixels"]:
                pixel.segments.remove(seg)
            ind = self.segments.index(seg)
            self.segments.remove(seg)
            new_segs = self.traverse(
                start=seg["pixels"][0], stop=seg["pixels"][-1])
            for new_seg in new_segs:
                self.segments.insert(ind, new_seg)
            # self.segments.extend(new_segs)

        if len(first.segments) > 0:
            seg = first.segments[0]
            first_ind = self.segments.index(seg)
            self.segments.remove(seg)
            for pixel in seg["pixels"]:
                pixel.segments.remove(seg)
            start = seg["pixels"][0]
        else:
            start = first
            first_ind = seg_ind

        # stop = False
        if len(last.segments) > 0:
            seg = last.segments[0]
            self.segments.remove(seg)
            for pixel in seg["pixels"]:
                pixel.segments.remove(seg)
            stop = seg["pixels"][-1]
        else:
            stop = last

        if start and stop:
            new_segments = self.traverse(start=start, stop=stop)
        # self.display.refresh()
        for seg in new_segments:
            self.segments.insert(first_ind, seg)
        # self.segments.extend(new_segments)
        # print_segments(self.segments)
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
    current = False
    for i in range(4):
        for segment in segments:
            pixels = segment["pixels"]
            vertical = segment["vertical"]
            direction = segment["direction"]
            if i == 0:
                if not vertical and direction[1] == 1:
                    current = True
                else:
                    current = False
            elif i == 1:
                if not vertical and direction[1] == -1:
                    current = True
                else:
                    current = False
            elif i == 2:
                if vertical and direction[0] == -1:
                    current = True
                else:
                    current = False
            else:
                if vertical and direction[0] == 1:
                    current = True
                else:
                    current = False
            if current:
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
    # im, scale = Image.open('contrast_micro.jpg'), 2
    # im, scale = Image.open('contrast_tiny.jpg'), 1
    # im, scale = Image.open('contrast_small.jpg'), 1
    im, scale = Image.open('contrast.jpg'), 1
    # im, scale = Image.open('face_tiny.jpg'), 1

    dithered = im.convert('1')
    source = np.array(dithered)
    drawing = Drawing(source)
    root = Tk()
    display = Display(root, drawing, scale=scale, ext=True)
    drawing.display = display
    # display.refresh()

    change = drawing.step()
    while change:
        # keyboard.wait('space')
        change = drawing.step()
        # display.refresh()

        # if drawing.counter > 200:
        #     drawing.counter = 0
        #     display.refresh()
        # else:
        #     drawing.counter += 1

    end_time = time.time()
    print("Time: ", int(end_time-start_time))
    drawing.save('test.png')
    display.refresh()
    root.mainloop()


if __name__ == '__main__':
    main()

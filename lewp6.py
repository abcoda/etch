from tkinter import Tk, Canvas, Frame, BOTH
from PIL import Image
import numpy as np

import keyboard
import time
import matplotlib.pyplot as plt
import csv
import random
# random.seed(0)


class Display(Frame):
    """
    Display: Class for a UI Window which displays the drawing in its current state.
    Each display object is linked to a drawing.
    Mainly for debugging purposes.
    """

    def __init__(self, root, drawing, scale=1, orig=True, n=2):
        super().__init__()

        self.root = root
        self.initUI()
        self.scale = scale
        self.drawing = drawing
        self.n = n
        self.offset = (2, 2)
        self.w = self.drawing.w * self.n * \
            self.scale + self.offset[0]*self.scale
        self.h = self.drawing.h * self.n * \
            self.scale + self.offset[1]*self.scale

        self.root.geometry(
            f"{self.w}x{self.h}+0+0")

    def initUI(self):
        self.master.title("Loop")
        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self)
        self.canvas.pack(fill=BOTH, expand=1)

    # Adds a single filled in pixel to the canvas
    def add(self, loc, color="black", width=0):
        self.canvas.create_rectangle(loc[0] * self.scale * self.n + self.offset[0]*self.scale, loc[1] * self.scale * self.n + self.offset[1]*self.scale, loc[0]
                                     * self.scale * self.n + self.scale + self.offset[0]*self.scale, loc[1] * self.scale * self.n + self.scale + self.offset[1]*self.scale, fill=color, width=width)

    def update(self, removed_pixels, added_pixels):
        for pixel in removed_pixels:
            loc = pixel["loc"]
            vect_future = pixel["vect_future"]
            vect_past = pixel["vect_past"]
            for i in range(self.n):
                self.add((loc[0] + vect_future[0]*i/self.n, loc[1] +
                          vect_future[1]*i/self.n), color="white")
                self.add((loc[0] - vect_past[0]*i/self.n, loc[1] -
                          vect_past[1]*i/self.n), color="white")
        for pixel in added_pixels:
            loc = pixel["loc"]
            vect_future = pixel["vect_future"]
            vect_past = pixel["vect_past"]
            for i in range(self.n):
                self.add((loc[0] + vect_future[0]*i/self.n, loc[1] +
                          vect_future[1]*i/self.n), color="black")
                self.add((loc[0] - vect_past[0]*i/self.n, loc[1] -
                          vect_past[1]*i/self.n), color="black")

        # self.root.update()
        return True

    def load(self):
        # Updates the canvas with the current state of the drawing

        self.canvas.create_rectangle(0, 0, self.w, self.h, fill='white')
        for row in range(self.drawing.h):
            for col in range(self.drawing.w):
                # Fill in pixels with different colors depending on the status of the pixel
                pixel = self.drawing.data[row][col]
                loc = (col, row)
                if pixel.inside == None:
                    future_loc = pixel.future.loc
                    vect = (future_loc[0] - loc[0], future_loc[1] - loc[1])
                    for i in range(self.n):
                        self.add((loc[0] + vect[0]*i/self.n,
                                  loc[1] + vect[1]*i/self.n), color="black")
                    self.add(loc, color='black')
                if pixel.anchor:
                    self.add(loc, color='red')
                elif pixel.inside:
                    self.add(loc, color='blue')
                elif pixel.inside == False:
                    self.add(loc, color='green')
                # elif col % 2 == row % 2:
                    # self.add(loc, color='#ddd')
                # else:
                #     self.add(loc, color='#eee')
        self.root.update()
        return True


class Pixel:
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


class Drawing:
    """
    Represents the Entire Drawing, including all of the pixel data, loop data, 
    and list of segments in the loop
    The source passed in to the drawing is a numpy array of 1's and 0's 
    coming from a strictly black and white image (see main)
    """

    def __init__(self, source, display=None):
        # counter counts the number of iterations of the step function, for debugging or animation purposes
        self.counter = 0
        self.h, self.w = source.shape
        self.display = display
        self.segments = []

        # Data is a 2D numpy array of pixel objects. Black pixels from the source become anchor points
        self.data = []
        self.anchor_count = 0
        self.num_anchors = 0
        for j in range(self.h):
            self.data.append([])
            for i in range(self.w):
                self.data[j].append(Pixel(
                    (i, j),
                    anchor=not source[j][i]
                ))
################################
        data = self.data

        x_max = []
        for i in range(self.w-1, -1, -1):
            for j in range(self.h):
                if data[j][i].anchor:
                    x_max.append(data[j][i])
            if len(x_max) > 0:
                break

        y_min = []
        for j in range(self.h):
            for i in range(self.w):
                if data[j][i].anchor:
                    y_min.append(data[j][i])
            if len(y_min) > 0:
                break

        head = y_min[0]
        head.past = None
        loop = []

        target_x = y_min[-1].x
        while head.x < target_x:
            loop.append(head)
            future = data[head.y][head.x + 1]
            head.future = future
            future.past = head
            head = future

        while head.x < x_max[0].x:
            loop.append(head)
            future = data[head.y + 1][head.x]
            head.future = future
            future.past = head
            head = future

            row_max = False
            for pixel in data[head.y][head.x:]:
                if pixel.anchor:
                    row_max = pixel

            if row_max:
                while head.x < row_max.x:
                    loop.append(head)
                    future = data[head.y][head.x + 1]
                    head.future = future
                    future.past = head
                    head = future

        while head.y < x_max[-1].y:
            loop.append(head)
            future = data[head.y + 1][head.x]
            head.future = future
            future.past = head
            head = future

        if head.y < self.h - 1:
            loop.append(head)
            future = data[head.y + 1][head.x]
            head.future = future
            future.past = head
            head = future

        x_max = False
        for i in range(head.x, -1, -1):
            for j in range(head.y + 1, self.h):
                if data[j][i].anchor:
                    x_max = data[j][i]
            if x_max:
                break

        while x_max:
            while head.x > x_max.x:
                loop.append(head)
                future = data[head.y][head.x-1]
                head.future = future
                future.past = head
                head = future
            while head.y < x_max.y:
                loop.append(head)
                future = data[head.y+1][head.x]
                head.future = future
                future.past = head
                head = future
            x_max = False
            for i in range(head.x, -1, -1):
                for j in range(head.y + 1, self.h):
                    if data[j][i].anchor:
                        x_max = data[j][i]
                if x_max:
                    break

        x_min = False
        for pixel in data[head.y]:
            if pixel.anchor:
                x_min = pixel
                break

        while head.x > x_min.x:
            loop.append(head)
            future = data[head.y][head.x - 1]
            head.future = future
            future.past = head
            head = future

        if head.x > 0:
            loop.append(head)
            future = data[head.y][head.x - 1]
            head.future = future
            future.past = head
            head = future

        # loop.append(head)
        # future = data[head.y-1][head.x]
        # head.future = future
        # future.past = head
        # head = future

        x_min = []
        for i in range(self.w):
            for j in range(self.h):
                if data[j][i].anchor:
                    x_min.append(data[j][i])
            if len(x_min) > 0:
                break

        while head.x > x_min[-1].x:  # chaned
            loop.append(head)
            future = data[head.y - 1][head.x]
            head.future = future
            future.past = head
            head = future

            row_min = False
            for pixel in data[head.y][:head.x]:
                if pixel.anchor:
                    row_min = pixel
                    break

            if row_min:
                while head.x > row_min.x:
                    loop.append(head)
                    future = data[head.y][head.x - 1]
                    head.future = future
                    future.past = head
                    head = future

        while head.y > x_min[0].y:
            loop.append(head)
            future = data[head.y - 1][head.x]
            head.future = future
            future.past = head
            head = future

        if head.y > 0:
            loop.append(head)
            future = data[head.y - 1][head.x]
            head.future = future
            future.past = head
            head = future

        x_min = False
        for i in range(head.x, self.w-1):
            for j in range(head.y):
                if data[j][i].anchor:
                    x_min = data[j][i]
            if x_min:
                break

        while x_min:
            while head.x < x_min.x:
                loop.append(head)
                future = data[head.y][head.x+1]
                head.future = future
                future.past = head
                head = future
            while head.y > x_min.y:
                loop.append(head)
                future = data[head.y-1][head.x]
                head.future = future
                future.past = head
                head = future
            x_min = False
            for i in range(head.x, self.w-1):
                for j in range(head.y - 1, -1, -1):
                    if data[j][i].anchor:
                        x_min = data[j][i]
                if x_min:
                    break

        for row in data:
            for pixel in row:
                pixel.inside = True

        for pixel in loop:
            pixel.inside = None

        for row in data:
            x_min = self.w
            x_max = 0
            for pixel in row:
                if pixel.inside == None:
                    if pixel.x < x_min:
                        x_min = pixel.x
                    if pixel. x > x_max:
                        x_max = pixel.x
            for pixel in row:
                if pixel.x < x_min:
                    pixel.inside = False
                elif pixel.x > x_max:
                    pixel.inside = False
                else:
                    pixel.inside = True

################################
        # # create initial loop along outer boundary
        # data = self.data
        # first = data[0][0]
        # past = first

        # # starting at the top left, work around the outside border clockwise, setting each pixels past and future
        # for i in range(self.w):
        #     current = data[0][i]
        #     current.past = past
        #     current.past.future = current
        #     past = current
        # for j in range(1, self.h):
        #     current = data[j][self.w - 1]
        #     current.past = past
        #     current.past.future = current
        #     past = current
        # for i in range(self.w - 2, -1, -1):
        #     current = data[self.h - 1][i]
        #     current.past = past
        #     current.past.future = current
        #     past = current
        # for j in range(self.h-2, 0, -1):
        #     current = data[j][0]
        #     current.past = past
        #     current.past.future = current
        #     past = current
        # current.future = first
        # first.past = current

        # # mark all pixels contained inside the loop as "inside", and outside as "not inside".
        # # Loop pixels have None value for "inside"
        # for j in range(1, self.h - 1):
        #     for i in range(1, self.w - 1):
        #         self.data[j][i].inside = True

        # # the loop list holds references to all of the pixels that are currently part of the loop
        # loop = []
        # pixel = self.data[0][0]
        # while True:
        #     loop.append(pixel)
        #     pixel = pixel.future
        #     if pixel == loop[0]:
        #         break

        #########################################################
        self.loop = loop
        for row in data:
            for pixel in row:
                if pixel.anchor:
                    self.num_anchors += 1
                    if pixel.inside == None:
                        self.anchor_count += 1

        for pixel in loop:
            pixel.inside = None

        # get the initial segments of the loop by traversing the entire loop
        self.segments = self.traverse()

    def get_direction(self, pixel, vertical):
        """Given a pixel's coordinates (x,y) and the orientation of the containing segment,
        get a vector of the direction towards the inside of the loop.
        Returns None if none of the neighboring pixels are inside (corners)"""
        x, y = pixel.loc
        # if the segment is vertical, just look left and right of the pixel
        if vertical:
            try:
                if self.data[y][x + 1].inside:
                    return (1, 0)
            except:
                pass
            try:
                if self.data[y][x-1].inside:
                    return (-1, 0)
            except:
                pass
        # if the segment is horizontal, just look above and below the pixel
        else:
            try:
                if self.data[y-1][x].inside:
                    return (0, -1)
            except:
                pass
            try:
                if self.data[y+1][x].inside:
                    return (0, 1)
            except:
                pass
        return None

    def traverse(self, start=False, stop=False):
        """
        Traverse a portion of the loop (or entire loop if no start or finish are supplied), 
        determining the segments that make up the traversed section.
        Only valid segments are recorded (single pixel segments do not count, and a segment must
        be able to be advanced by at least one pixel without hitting any other pixels)
        """
        # if no start is given, start at the first pixel in the loop list (arbitrary)
        if not start:
            start = self.loop[0]
        head = start  # 'head' is the current pixel at any time as the loop is traversed
        # Anchor points are part of two segments so must be considered twice while traversing.
        # anchor_flag denotes whether the anchor has already been considered once
        # If the first pixel is an anchor, only consider it once (until the end)
        if start.anchor:
            anchor_flag = True
        else:
            anchor_flag = False
        if not stop:
            stop = start
            full_loop = True
        else:
            full_loop = False
            stop = stop.future
        stop_flag = False

        segments = []
        while True:
            if stop_flag:
                break

            vertical = head.future.x == head.x
            direction = self.get_direction(head, vertical)
            pixels = []
            while True:
                # add the current pixel to the segment if there is an empty inside neighbor
                if self.get_direction(head, vertical) is not None:
                    pixels.append(head)
                    # if we are at an anchor or a corner
                    if head.anchor or (head.future.x == head.x) != vertical:
                        if anchor_flag:
                            anchor_flag = False
                        # if this is the first time at the anchor, end the segment here and start a new one
                        else:
                            anchor_flag = True
                            break
                    else:
                        anchor_flag = False
                    # advance the head, and if we're at the stopping point, end the traversal
                    # unless the whole loop is being traversed, then finish the current segment first
                    head = head.future
                    if head is stop:
                        stop_flag = True
                        if not full_loop:
                            break
                # if the pixel is NOT valid, end the segment and advance (unless we are at an anchor for the first time)
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
                    head = head.future
                    if head is stop:
                        stop_flag = True
                    break

            # if the segment is longer than one pixel, add it to the list of segments
            if len(pixels) > 1:
                segment = {
                    "pixels": pixels,
                    "vertical": vertical,
                    "direction": direction
                }
                # for each pixel in the segment, add the segment to the pixels interal list of its own segments
                segments.append(segment)
                for pixel in pixels:
                    if segment not in pixel.segments:
                        pixel.segments.append(segment)

        # this function only returns the segments, but relies on the calling function to delete the
        # old segments for this section of the loop
        return segments

    def step(self, compress=False):
        segments = self.segments
        # select a segment to step forward
        while True:
            try:
                # first find the longest segment (or one of the longest segments)
                segment = max(
                    segments, key=lambda segment: len(segment["pixels"]))

                # if there are multiple segments with the max length, we can randomly choose one to make
                # the final product less orderly looking. Comment these lines out for a different look
                length = len(segment["pixels"])
                choices = [segment for segment in segments if len(
                    segment["pixels"]) == length]
                segment = random.choice(choices)
            # if there are no available segments to move, the drawing is complete
            except:
                print("\nNo segments")
                return False, False
            # if the compress parameter is true, the loop closes in on itself as much as possible, even
            # when unnecessary for reaching anchor points
            if compress:
                break
            # if we don't want to compress the loop, we should only select segments that can actually
            # reach anchor points when stepped in the given direction
            direction = segment["direction"]
            pixels = segment["pixels"][:]
            target_found = False
            failed = []
            # keep looking one row/column into the future until we reach an anchor or an obstacle
            # if we do encounter an obstacle, just remove the offending pixel and keep checking,
            # becuase it could still be possible to reach an anchor with a sub-segment
            d = 1
            while len(pixels) > 0:
                for pixel in pixels:
                    future = self.data[pixel.y + d *
                                       direction[1]][pixel.x + d*direction[0]]
                    if future.inside == True:
                        if future.anchor:
                            target_found = True
                            break
                    else:
                        failed.append(pixel)
                for pixel in failed:
                    pixels.remove(pixel)
                failed = []
                d += 1
            # if there is a reachable anchor point, continue with the current segment
            if target_found:
                break
            # if no anchors can be reached by the segment, delete the segment and try another one
            else:
                for pixel in segment["pixels"]:
                    pixel.segments.remove(segment)
                segments.remove(segment)

        # save the index of the selected segment for future reference
        seg_ind = self.segments.index(segment)
        vect = segment["direction"]

        # after advancing the segment, the first pixel of the segment will be in new_first_loc
        #       =x=====x=     ->   =x     x=
        #                           =======

        # reassign past and future references
        first = segment["pixels"][0]
        new_first = self.data[first.loc[1] + vect[1]][first.loc[0] + vect[0]]
        new_first.past = first
        first.future = new_first
        self.data[new_first.y][new_first.x].inside = None
        self.loop.append(new_first)
        first.segments.remove(segment)
        # new_pixels = [new_first, first]
        new_pixels = [new_first]

        removed_pixels = []
        for pixel in segment["pixels"]:
            loc = pixel.loc
            future_loc = pixel.future.loc
            past_loc = pixel.past.loc
            future_direction = (future_loc[0] - loc[0], future_loc[1] - loc[1])
            past_direction = (loc[0] - past_loc[0], loc[1] - past_loc[1])
            removed_pixels.append(
                {"loc": loc, "vect_future": future_direction, "vect_past": past_direction})

        # for all of the pixels in between the first and last,
        #    just advance them towards the inside without changing past or future
        # 'flagged' is a list of any pixels whose segments might be affected by the current step
        #     (moveing one segment forward could block a completely different segment from progressing)
        flagged = []
        past = new_first
        for pixel in segment["pixels"][1:-1]:
            new_pixel = self.data[pixel.loc[1] +
                                  vect[1]][pixel.loc[0] + vect[0]]
            new_pixels.append(new_pixel)
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
        new_pixels.append(new_last)
        # new_pixels.append(last)

        # remove the current segment (it's details are no longer accurate)
        self.segments.remove(segment)

        # check the four pixels that are against the new corners
        #      ===      ===
        #       x|=====|x
        #        x    x
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

        # go through all the flagged pixels and get all segments that they are a part of
        redo_segments = []
        for pixel in flagged:
            for seg in pixel.segments:
                if seg not in redo_segments:
                    redo_segments.append(seg)
        # for each segment, remove it from its pixels internal list (it needs to be updated now)
        for seg in redo_segments:
            for pixel in seg["pixels"]:
                pixel.segments.remove(seg)
            ind = self.segments.index(seg)
            self.segments.remove(seg)
            # get the updated segments by traversing over the original segments' start to finish
            new_segs = self.traverse(
                start=seg["pixels"][0], stop=seg["pixels"][-1])
            # add the updated segment(s) the the master list of segements
            # by inserting at the same index, the loop progresses  in a more systematic way (for animation)
            for new_seg in new_segs:
                self.segments.insert(ind, new_seg)

        # if the first pixel was part of another segment, that segment could have been affected as well,
        # so we need to traverse that segment as well
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

        # similarly, if the last pixel is part of another segment, traverse that one as well
        if len(last.segments) > 0:
            seg = last.segments[0]
            self.segments.remove(seg)
            for pixel in seg["pixels"]:
                pixel.segments.remove(seg)
            stop = seg["pixels"][-1]
        else:
            stop = last

        # traverse the new segment, including the previous and following segments if needed
        new_segments = self.traverse(start=start, stop=stop)
        for seg in new_segments:
            self.segments.insert(first_ind, seg)

        added_pixels = []
        for pixel in new_pixels:
            loc = pixel.loc
            future_loc = pixel.future.loc
            past_loc = pixel.past.loc
            future_direction = (future_loc[0] - loc[0], future_loc[1] - loc[1])
            past_direction = (loc[0] - past_loc[0], loc[1] - past_loc[1])
            added_pixels.append(
                {"loc": loc, "vect_future": future_direction, "vect_past": past_direction})
            if pixel.anchor:
                self.anchor_count += 1

        return removed_pixels, added_pixels

    # simply checks whether an (x,y) location is within the bounds of the drawing
    def is_valid_loc(self, loc):
        x, y = loc
        return x >= 0 and y >= 0 and x < self.w and y < self.h

    # returns a list of (x,y) coordinates of a pixel's neighbors (only ones that are valid locations)
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

    def save(self, name, n=2):
        # save the drawing as an image file
        # n sets the expansion factor (n of at least 2 must be used to visibly see the loop)

        # offset everything down and to the right by one pixel (so the loop is centered)
        im_data = np.ones((self.h*n + 1, self.w*n + 1),
                          dtype='bool')  # initially all white

        for pixel in self.loop:
            # determine the direction of the next pixel, so that the gaps can be filled in properly
            loc = pixel.loc
            future_loc = pixel.future.loc
            vect = (future_loc[0]-loc[0], future_loc[1]-loc[1])

            for i in range(n):
                im_data[1 + pixel.y*n + vect[1]*i][1 +
                                                   pixel.x*n + vect[0]*i] = False  # False <=> Black

        im = Image.fromarray(im_data)
        im.save(name, format='png')


def print_segments(segments):
    # print out a list of segments (for debugging)
    # groups segments by orientation and direction
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


def loop(im, show=False, scale=1, compress=True, timer=True, filename='latest.png', n=2):
    start_time = time.time()
    dithered = im.convert('1')
    # dithered.show()
    source = np.array(dithered)
    drawing = Drawing(source)
    if show:
        root = Tk()
        display = Display(root, drawing, scale=scale, n=n)
        drawing.display = display
        display.load()
        # display.root.update()
        # drawing.save('test.png')
        # time.sleep(5000)

    # time.sleep(1000)
    name_counter = 0
    removed_pixels, added_pixels = drawing.step(compress=compress)
    while added_pixels:
        # anchor_count = 0
        # for pixel in drawing.loop:
        # if pixel.anchor:
        # anchor_count += 1
        print(f"\r -- {drawing.anchor_count}/{drawing.num_anchors}", end='')

        if show:
            # keyboard.wait('space')
            # display.update(removed_pixels, added_pixels)
            # display.root.update()
            if drawing.counter > 8:
                drawing.counter = 0
            # display.root.update()
                # drawing.save(f'elliot2/{name_counter}.png')
                name_counter += 1
            else:
                drawing.counter += 1

        removed_pixels, added_pixels = drawing.step(compress=compress)

    end_time = time.time()
    if timer:
        print("Time: ", int(end_time-start_time))
        print("Loop Length: ", len(drawing.loop))
    drawing.save(filename, n=n)
    if show:
        root.mainloop()


def main():
    # im, scale = Image.open('images/sliver.jpg'), 10
    # im, scale = Image.open('images/contrast_micro.jpg'), 2
    # im, scale = Image.open('images/contrast_tiny.jpg'), 1
    # im, scale = Image.open('images/contrast_small.jpg'), 1
    # im, scale = Image.open('images/contrast.jpg'), 1
    # im, scale = Image.open('images/dith_line_02.png'), 1
    # im, scale = Image.open('images/rac_small.jpg'), 1
    # im, scale = Image.open('images/dithered.png'), 1
    # im, scale = Image.open('images/face_tiny.jpg'), 1
    # im, scale = Image.open('images/kramer_edited.jpg'), 1
    # im, scale = Image.open('images/sargent_0.jpg'), 1
    # im, scale = Image.open('images/d2.jpg'), 1
    im, scale = Image.open('images/angela.png'), 1

    # im, scale = Image.open('images/tritone.jpg'), 1
    # im, scale = Image.open('images/grid.png'), 1

    # loop(im, scale=scale, compress=True, show=True, n=2)
    loop(im, scale=scale, compress=True, show=False, n=2)


if __name__ == '__main__':
    main()

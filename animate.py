from tkinter import Tk, Canvas, Frame, BOTH
import csv
from PIL import Image
from get_path import get_path


class Display(Frame):

    def __init__(self, root, dim, scale=1):
        super().__init__()

        self.root = root
        self.initUI()
        self.scale = scale
        self.root.geometry(
            f"{dim[0] * self.scale}x{dim[1] * self.scale}+0+0")

    def initUI(self):
        self.master.title("Loop")
        self.pack(fill=BOTH, expand=1)
        self.canvas = Canvas(self)
        self.canvas.pack(fill=BOTH, expand=1)

    # Adds a single filled in pixel to the canvas
    def add(self, loc, color="black", width=0):
        self.canvas.create_rectangle(loc[0] * self.scale, loc[1] * self.scale, loc[0]
                                     * self.scale + self.scale, loc[1] * self.scale + self.scale, fill=color, width=width)


def animate(im='latest.png'):
    im = Image.open(im)
    root = Tk()
    w, h = im.size
    display = Display(root, (w, h), scale=1)
    path = get_path(im)
    counter = 0
    for loc in path:
        display.add(loc)
        counter += 1
        if counter > 50:
            root.update()
            counter = 0
    root.mainloop()


def main():
    animate()


if __name__ == '__main__':
    main()

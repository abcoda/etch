from tkinter import Tk, Canvas, Frame, BOTH
import csv


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


def main():
    root = Tk()
    w = 829
    h = 467
    display = Display(root, (w, h), scale=1)
    with open('path_data.csv', newline='') as csvfile:
        reader = csv.reader(csvfile)
        counter = 0
        for row in reader:
            loc = (int(row[0]), int(row[1]))
            display.add(loc)
            counter += 1
            if counter > 50:
                root.update()
                counter = 0
    root.mainloop()


if __name__ == '__main__':
    main()

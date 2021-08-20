from PIL import Image
import csv
import numpy as np
from get_path import get_path


def get_arduino_path(im, n=2):
    data = np.array(im, dtype='bool')
    h, w = data.shape
    x, y = False, False
    for row in range(h):
        for col in range(w):
            if not data[row][col]:
                x, y = col, row
                break
        if x:
            break
    path = ""
    start = (x, y)
    direct = 0
    while True:
        if not data[y][x+1] and direct != 1:
            x += 1
            direct = 3
            path += '00'
        elif not data[y+1][x] and direct != 2:
            y += 1
            direct = 4
            path += '10'
        elif not data[y][x-1] and direct != 3:
            x -= 1
            direct = 1
            path += '01'
        elif not data[y-1][x] and direct != 4:
            y -= 1
            direct = 2
            path += '11'
        if (x, y) == start:
            break
    return path


def main():
    im = Image.open('latest.png')
    path = get_arduino_path(im)
    with open('arduino_path.txt', 'w') as f:
        f.write(path)


if __name__ == '__main__':
    main()

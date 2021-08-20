from PIL import Image
import csv
import numpy as np


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
    path = []
    direct = 0
    while True:
        if not data[y][x+1] and direct != 1:
            x += 1
            direct = 3
        elif not data[y+1][x] and direct != 2:
            y += 1
            direct = 4
        elif not data[y][x-1] and direct != 3:
            x -= 1
            direct = 1
        elif not data[y-1][x] and direct != 4:
            y -= 1
            direct = 2
        path.append((x, y))
        if (x, y) == path[0]:
            break
    return path


def main():
    im = Image.open('latest.png')
    path = get_path(im, n=2)
    with open('path_data.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        for pixel in path:
            writer.writerow(pixel)


if __name__ == '__main__':
    main()

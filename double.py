from PIL import Image
from sys import argv
import numpy as np

im = Image.open(argv[1])
n = int(argv[2])

source = np.array(im)
# print(source)
if len(source.shape) > 2:
    h, w, _ = source.shape
    copy = np.ones((h, w))
    for row in range(h):
        for col in range(w):
            copy[row][col] = source[row][col][0]
    source = copy
else:
    h, w = source.shape
# print(source)
big = np.ones((h*n, w*n), dtype='bool')
for row in range(h):
    for col in range(w):
        # print(row, col)
        for i in range(n):
            for j in range(n):
                big[row*n + j][col*n + i] = source[row][col]
# print(big)
big = Image.fromarray(big)
big.show()

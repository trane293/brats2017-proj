from PIL import Image
import matplotlib.pyplot as plt
import numpy as np


def prepImage(image='../test_cases/easy.png'):
    img = Image.open(image).convert('L')
    img = np.asarray(img)
    img2 = np.copy(img)

    img2[np.where(np.logical_and(np.greater_equal(img, 91), np.less_equal(img, 168)))] = 2.0
    img2[np.where(img == 255)] = 1.0

    return img2


class Formatter(object):
    '''
    Use like:
        fig, ax = plt.subplots()
        im = ax.imshow(img, cmap='gray')
        ax.format_coord = Formatter(im)
    '''

    def __init__(self, im):
        self.im = im

    def __call__(self, x, y):
        z = self.im.get_array()[int(y), int(x)]
        return 'x={:.01f}, y={:.01f}, z={:.01f}'.format(x, y, z)

img = prepImage(image='../test_cases/e2.png')
fig, ax = plt.subplots()
im = ax.imshow(img, cmap='gray')
ax.format_coord = Formatter(im)
plt.show()

testcase = np.array(
    [[0,3,3,0],
    [0,3,3,0],
    [0,2,2,0],
    [0,2,2,0],
    [0,2,2,0],
    [0,1,1,0],
    [0,1,1,0]]
)

from networkx import DiGraph
g = DiGraph()
# self links
g.add_edge(0,0)
g.add_edge(1,1)
g.add_edge(2,2)
g.add_edge(3,3)
g.add_edge(1,0)
g.add_edge(1,3)
g.add_edge(1,2)
g.add_edge(2,0)
g.add_edge(2,3)
g.add_edge(3,0)

from scipy.ndimage.measurements import center_of_mass


def maskExcept(im, e=1):
    tm = np.array(im)
    tm[np.where(tm != e)] = 0
    return tm


core = maskExcept(testcase, e=1)  # get only the necrotic core

if np.count_nonzero(core) == 0:
    print('No necrotic core found, reverting to enhancing!')
    core = maskExcept(testcase, e=2)  # get the enhancing core

comr, comc = center_of_mass(core)
comr = int(comr)
comc = int(comc)

names = {
    1: 'necrotic',
    2: 'enhancing',
    3: 'edema'
}

# start the scanning process
switch = []

# direction: up. We will change the rows
prev = None
flag = 1
for coord in range(int(comr), -1, -1):
    r = coord
    c = comc

    if prev == None:
        prev = testcase[r, c]
    else:
        if g.has_edge(prev, testcase[r, c]):
            prev = testcase[r, c]
            continue
        else:
            print('invalid!')
            flag = -1
            break

if flag != -1:
    print('Passed the Up direction test!')
# direction: down
# direction: down
prev = None
flag = 1
for coord in range(int(comr), np.shape(testcase)[0], 1):
    r = coord
    c = comc

    if prev == None:
        prev = testcase[r, c]
    else:
        if g.has_edge(prev, testcase[r, c]):
            prev = testcase[r, c]
            continue
        else:
            print('invalid!')
            flag = -1

if flag != -1:
    print('Passed the Down direction test!')
# direction: right




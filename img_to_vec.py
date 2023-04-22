import math
from cv2 import Mat
import cv2
import numpy as np

COLOR_PALETTE = [
    # Yellow
    {'name': 0, 'values': [219, 180, 4]},
    {'name': 1, 'values': [226, 163, 27]},
    {'name': 2, 'values': [223, 193, 89]},
    {'name': 3, 'values': [215, 191, 85]},
    {'name': 4, 'values': [220, 198, 94]},

    # White
    {'name': 5, 'values': [214, 211, 215]},

    # Brown
    {'name': 6, 'values': [160, 74, 10]},
    {'name': 7, 'values': [140, 97, 66]},
    {'name': 8, 'values': [108, 67, 31]},
    {'name': 9, 'values': [92, 57, 40]},
    {'name': 10, 'values': [76, 37, 10]},
    {'name': 11, 'values': [134, 62, 2]},

    # Lightblue
    {'name': 12, 'values': [78, 175, 199]},
    {'name': 13, 'values': [55, 159, 194]},
    {'name': 14, 'values': [133, 190, 201]},
    {'name': 15, 'values': [126, 186, 202]},
    {'name': 16, 'values': [116, 185, 196]},

    # Blue
    {'name': 17, 'values': [61, 73, 166]},
    {'name': 18, 'values': [151, 171, 211]},
    {'name': 19, 'values': [53, 103, 149]},
    {'name': 20, 'values': [46, 89, 107]},
    {'name': 21, 'values': [19, 80, 177]},

    # Green
    {'name': 22, 'values': [66, 167, 12]},
    {'name': 23, 'values': [24, 49, 0]},
    {'name': 24, 'values': [100, 138, 71]},
    {'name': 25, 'values': [0, 117, 95]},

    # Red
    {'name': 26, 'values': [157, 56, 13]},
    {'name': 27, 'values': [112, 33, 49]},
    {'name': 28, 'values': [168, 0, 12]},
    {'name': 29, 'values': [103, 26, 45]},

    # Lightbrown
    {'name': 30, 'values': [200, 165, 117]},
    {'name': 31, 'values': [185, 134, 87]},

    # Pink
    {'name': 32, 'values': [210, 111, 98]},
    {'name': 33, 'values': [209, 159, 180]},
    {'name': 34, 'values': [211, 112, 99]},
    {'name': 35, 'values': [205, 121, 145]},

    # Purple
    {'name': 36, 'values': [169, 108, 168]},
    {'name': 37, 'values': [147, 74, 155]},
    {'name': 38, 'values': [152, 117, 173]},
    {'name': 39, 'values': [76, 55, 113]},
    {'name': 40, 'values': [131, 121, 179]},

    # Lightgrey
    {'name': 41, 'values': [171, 189, 217]},
    {'name': 42, 'values': [194, 195, 198]},

    # Grey
    {'name': 43, 'values': [112, 116, 120]},
    {'name': 44, 'values': [159, 164, 177]},
    {'name': 45, 'values': [154, 139, 129]},
    {'name': 46, 'values': [161, 158, 162]},
    {'name': 47, 'values': [181, 182, 185]},

    # Black
    {'name': 48, 'values': [48, 28, 21]},
    {'name': 49, 'values': [3, 31, 20]},
    {'name': 50, 'values': [45, 22, 0]},
    {'name': 51, 'values': [1, 1, 1]},

    # Orange
    {'name': 52, 'values': [230, 160, 18]},
    {'name': 53, 'values': [231, 159, 17]},
    {'name': 54, 'values': [218, 164, 32]},
    {'name': 55, 'values': [186, 78, 13]},
]

MESH_SIZE = 4


def image_to_vec(filename: str) -> list[int]:
    vector = [0] * COLOR_PALETTE.__len__()

    for image in split(filename):
        vector = np.add(vector, child_image_to_vec(image))

    return vector.tolist()


def child_image_to_vec(img: Mat) -> list[int]:
    vector = [0] * COLOR_PALETTE.__len__()
    for i in range(img.shape[0]):
        vec = img[1: 2, i, :][0]
        vector[map_image_color_to_palette_color(vec)] += 1

    return vector


def map_image_color_to_palette_color(vector: list[int]) -> int:
    distanceResult = 0
    colorResult = {}
    for color in COLOR_PALETTE:
        p_distance = distance(vector, color['values'])
        if p_distance < distanceResult or colorResult == {}:
            distanceResult = p_distance
            colorResult = color

    return colorResult['name']


def distance(element_1: list[int], element_2: list[int]) -> float:
    result = 0
    for i in range(len(element_1)):
        result = result + math.pow(element_1[i] - element_2[i], 2)

    return math.sqrt(result)


def split(filename: str) -> list[Mat]:
    list = []
    img1 = cv2.imread(filename)
    img = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    w = int(img.shape[0]/MESH_SIZE)
    h = int(img.shape[1]/MESH_SIZE)

    for i in range(MESH_SIZE):
        x1 = i*w
        x2 = x1+w

        for j in range(MESH_SIZE):
            y1 = j*h
            y2 = y1+h
            list.append(img[x1:x2, y1:y2])

    return list

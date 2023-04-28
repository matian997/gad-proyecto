import math
from cv2 import Mat
import cv2
import numpy as np

COLOR_PALETTE = [
    # Yellow
    {'name': 0, 'values': [215, 191, 85]},
    {'name': 1, 'values': [219, 180, 4]},
    {'name': 2, 'values': [223, 193, 89]},
    {'name': 3, 'values': [220, 198, 94]},
    {'name': 4, 'values': [226, 163, 27]},

    # White
    {'name': 5, 'values': [179, 189, 184]},
    {'name': 6, 'values': [214, 211, 215]},
    {'name': 7, 'values': [214, 214, 214]},

    # Brown
    {'name': 8, 'values': [76, 37, 10]},
    {'name': 9, 'values': [92, 57, 40]},
    {'name': 10, 'values': [108, 67, 31]},
    {'name': 11, 'values': [134, 62, 2]},
    {'name': 12, 'values': [140, 97, 66]},
    {'name': 13, 'values': [160, 74, 10]},

    # Lightblue
    {'name': 14, 'values': [55, 159, 194]},
    {'name': 15, 'values': [60, 161, 202]},
    {'name': 16, 'values': [61, 134, 181]},
    {'name': 17, 'values': [78, 175, 199]},
    {'name': 18, 'values': [90, 175, 192]},
    {'name': 19, 'values': [116, 185, 196]},
    {'name': 20, 'values': [126, 186, 202]},
    {'name': 21, 'values': [133, 190, 201]},

    # Blue
    {'name': 22, 'values': [11, 127, 196]},
    {'name': 23, 'values': [19, 80, 177]},
    {'name': 24, 'values': [28, 3, 100]},
    {'name': 25, 'values': [32, 140, 198]},
    {'name': 26, 'values': [46, 89, 107]},
    {'name': 27, 'values': [49, 25, 115]},
    {'name': 28, 'values': [53, 103, 149]},
    {'name': 29, 'values': [61, 73, 166]},
    {'name': 30, 'values': [151, 171, 211]},

    # Green
    {'name': 31, 'values': [0, 117, 95]},
    {'name': 32, 'values': [2, 124, 110]},
    {'name': 33, 'values': [14, 72, 59]},
    {'name': 34, 'values': [24, 49, 0]},
    {'name': 35, 'values': [66, 167, 12]},
    {'name': 36, 'values': [100, 138, 71]},
    {'name': 37, 'values': [144, 155, 105]},
    {'name': 38, 'values': [158, 186, 72]},

    # Red
    {'name': 39, 'values': [103, 26, 45]},
    {'name': 40, 'values': [112, 33, 49]},
    {'name': 41, 'values': [122, 35, 26]},
    {'name': 42, 'values': [157, 56, 13]},
    {'name': 43, 'values': [168, 0, 12]},
    {'name': 44, 'values': [185, 55, 14]},
    {'name': 45, 'values': [198, 0, 20]},

    # Lightbrown
    {'name': 46, 'values': [185, 134, 87]},
    {'name': 47, 'values': [195, 165, 119]},
    {'name': 48, 'values': [200, 165, 117]},

    # Pink
    {'name': 49, 'values': [199, 89, 115]},
    {'name': 50, 'values': [205, 121, 145]},
    {'name': 51, 'values': [206, 111, 128]},
    {'name': 52, 'values': [209, 159, 180]},
    {'name': 53, 'values': [210, 111, 98]},
    {'name': 54, 'values': [211, 112, 99]},

    # Purple
    {'name': 55, 'values': [76, 55, 113]},
    {'name': 56, 'values': [88, 69, 122]},
    {'name': 57, 'values': [121, 96, 166]},
    {'name': 58, 'values': [122, 119, 168]},
    {'name': 59, 'values': [131, 121, 179]},
    {'name': 60, 'values': [147, 74, 155]},
    {'name': 61, 'values': [152, 117, 173]},
    {'name': 62, 'values': [169, 108, 168]},
    {'name': 63, 'values': [180, 176, 213]},

    # Lightgrey
    {'name': 64, 'values': [171, 164, 157]},
    {'name': 65, 'values': [171, 189, 217]},
    {'name': 66, 'values': [194, 195, 198]},

    # Grey
    {'name': 67, 'values': [112, 116, 120]},
    {'name': 68, 'values': [154, 139, 129]},
    {'name': 69, 'values': [159, 164, 177]},
    {'name': 70, 'values': [161, 158, 162]},
    {'name': 71, 'values': [181, 182, 185]},

    # Black
    {'name': 72, 'values': [1, 1, 1]},
    {'name': 73, 'values': [3, 31, 20]},
    {'name': 74, 'values': [8, 15, 13]},
    {'name': 75, 'values': [45, 22, 0]},
    {'name': 76, 'values': [48, 28, 21]},
    {'name': 77, 'values': [52, 53, 60]},

    # Orange
    {'name': 78, 'values': [186, 78, 13]},
    {'name': 79, 'values': [218, 164, 32]},
    {'name': 80, 'values': [230, 160, 18]},
    {'name': 81, 'values': [231, 159, 17]},
]

MESH_SIZE = 2
DEFAULT_IMAGE_WIDTH = 200
DEFAULT_IMAGE_HEIGHT = 200


def image_to_vec(filename: str) -> list[int]:
    vector = []

    for image in split(filename):
        vector += child_image_to_vec(image)

    return vector


def child_image_to_vec(img: Mat) -> list[int]:
    vector = [0] * COLOR_PALETTE.__len__()

    for subimg in img:
        for vec in subimg:
            vector[match_image_color_with_palette_color(vec)['name']] += 1

    return vector


def match_image_color_with_palette_color(vector: list[int]) -> any:
    distanceResult = 0
    colorResult = {}
    for color in COLOR_PALETTE:
        p_distance = distance(vector, color['values'])
        if (p_distance == 0):
            return color

        if p_distance < distanceResult or colorResult == {}:
            distanceResult = p_distance
            colorResult = color

    return colorResult


def distance(element_1: list[int], element_2: list[int]) -> float:
    result = 0
    for i in range(len(element_1)):
        result += math.pow(element_1[i] - element_2[i], 2)

    return math.sqrt(result)


def map_image_to_palette_color(filename: str):
    img = cv2.imread(filename)
    cv2.imshow('original_'+filename, img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mapped_image = np.zeros(img.shape, dtype=np.uint8)

    i = 0
    for subimg in img:
        j = 0
        for vec in subimg:
            mapped_image[i, j] = match_image_color_with_palette_color(vec)[
                'values']
            j += 1
        i += 1

    mapped_image = cv2.cvtColor(mapped_image, cv2.COLOR_RGB2BGR)
    cv2.imshow(filename, mapped_image)
    cv2.waitKey(0)


def split(filename: str) -> list[Mat]:
    list = []
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # if img.shape[0] != DEFAULT_IMAGE_HEIGHT:
    #     img = cv2.resize(img, dsize=[DEFAULT_IMAGE_HEIGHT, img.shape[1]])

    # if img.shape[1] != DEFAULT_IMAGE_WIDTH:
    #     img = cv2.resize(img, dsize=[img.shape[0], DEFAULT_IMAGE_WIDTH])

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

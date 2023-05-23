import colorsys
from enum import Enum
import time
from typing import Literal
from cv2 import Mat
import cv2
import numpy as np


class Color(Enum):
    RED = 0
    GREEN = 1
    BLUE = 2
    WHITE = 3
    GREY = 4
    BLACK = 5


COLOR_PALETTE = [
    {'values': Color.RED.value, 'children': [
        # brown
        {'name': 0, 'values': [76, 37, 10]},
        {'name': 1, 'values': [92, 57, 40]},
        {'name': 2, 'values': [108, 67, 31]},
        {'name': 3, 'values': [131, 114, 87]},
        {'name': 4, 'values': [134, 62, 2]},
        {'name': 5, 'values': [140, 97, 66]},
        {'name': 6, 'values': [160, 74, 10]},

        # red
        {'name': 7, 'values': [103, 26, 45]},
        {'name': 8, 'values': [112, 33, 49]},
        {'name': 9, 'values': [122, 35, 26]},
        {'name': 10, 'values': [157, 56, 13]},
        {'name': 11, 'values': [168, 0, 12]},
        {'name': 12, 'values': [185, 55, 14]},
        {'name': 13, 'values': [198, 0, 20]},

        # Lightbrown
        {'name': 14, 'values': [185, 134, 87]},
        {'name': 15, 'values': [195, 165, 119]},
        {'name': 16, 'values': [200, 165, 117]},

        # Pink
        {'name': 17, 'values': [199, 89, 115]},
        {'name': 18, 'values': [205, 121, 145]},
        {'name': 19, 'values': [206, 111, 128]},
        {'name': 20, 'values': [209, 159, 180]},
        {'name': 21, 'values': [210, 111, 98]},
        {'name': 22, 'values': [211, 112, 99]},

        # yellow

        {'name': 23, 'values': [215, 191, 85]},
        {'name': 24, 'values': [219, 180, 4]},
        {'name': 25, 'values': [223, 193, 89]},
        {'name': 26, 'values': [220, 198, 94]},
        {'name': 27, 'values': [226, 163, 27]},

        # Orange
        {'name': 28, 'values': [186, 78, 13]},
        {'name': 29, 'values': [218, 164, 32]},
        {'name': 30, 'values': [230, 160, 18]},
        {'name': 31, 'values': [231, 159, 17]},
    ]},
    {'values': Color.BLUE.value, 'children': [
        # Lightblue
        {'name': 32, 'values': [55, 159, 194]},
        {'name': 33, 'values': [60, 161, 202]},
        {'name': 34, 'values': [61, 134, 181]},
        {'name': 35, 'values': [78, 175, 199]},
        {'name': 36, 'values': [90, 175, 192]},
        {'name': 37, 'values': [116, 185, 196]},
        {'name': 38, 'values': [126, 186, 202]},
        {'name': 39, 'values': [133, 190, 201]},

        # Blue
        {'name': 40, 'values': [11, 127, 196]},
        {'name': 41, 'values': [19, 80, 177]},
        {'name': 42, 'values': [28, 3, 100]},
        {'name': 43, 'values': [32, 140, 198]},
        {'name': 44, 'values': [46, 89, 107]},
        {'name': 45, 'values': [49, 25, 115]},
        {'name': 46, 'values': [53, 103, 149]},
        {'name': 47, 'values': [61, 73, 166]},
        {'name': 48, 'values': [151, 171, 211]},

        # Purple
        {'name': 49, 'values': [76, 55, 113]},
        {'name': 50, 'values': [88, 69, 122]},
        {'name': 51, 'values': [121, 96, 166]},
        {'name': 52, 'values': [122, 119, 168]},
        {'name': 53, 'values': [131, 121, 179]},
        {'name': 54, 'values': [147, 74, 155]},
        {'name': 55, 'values': [152, 117, 173]},
        {'name': 56, 'values': [169, 108, 168]},
        {'name': 57, 'values': [180, 176, 213]},
    ]},
    {'values': Color.GREEN.value, 'children': [
        # green
        {'name': 58, 'values': [0, 117, 95]},
        {'name': 59, 'values': [2, 124, 110]},
        {'name': 60, 'values': [14, 72, 59]},
        {'name': 61, 'values': [24, 49, 0]},
        {'name': 62, 'values': [66, 167, 12]},
        {'name': 63, 'values': [100, 138, 71]},
        {'name': 64, 'values': [144, 155, 105]},
        {'name': 65, 'values': [158, 186, 72]},
        {'name': 84, 'values': [56, 169, 150]},

    ]},
    {'values': Color.WHITE.value, 'children': [
        # White
        {'name': 66, 'values': [179, 189, 184]},
        {'name': 67, 'values': [214, 211, 215]},
        {'name': 68, 'values': [214, 214, 214]},
    ]},
    {'values': Color.GREY.value, 'children': [
        # Lightgrey
        {'name': 69, 'values': [171, 164, 157]},
        {'name': 70, 'values': [171, 189, 217]},
        {'name': 71, 'values': [194, 195, 198]},

        # Grey
        {'name': 72, 'values': [105, 92, 82]},
        {'name': 73, 'values': [112, 116, 120]},
        {'name': 74, 'values': [154, 139, 129]},
        {'name': 75, 'values': [159, 164, 177]},
        {'name': 76, 'values': [161, 158, 162]},
        {'name': 77, 'values': [181, 182, 185]},
    ]},
    {'values': Color.BLACK.value, 'children': [
        # black
        {'name': 78, 'values': [1, 1, 1]},
        {'name': 79, 'values': [3, 31, 20]},
        {'name': 80, 'values': [8, 15, 13]},
        {'name': 81, 'values': [45, 22, 0]},
        {'name': 82, 'values': [48, 28, 21]},
        {'name': 83, 'values': [52, 53, 60]},
    ]}
]


MESH_SIZE = 3
DEFAULT_IMAGE_WIDTH = 200
DEFAULT_IMAGE_HEIGHT = 200
lastPaletteColor = {}
lastSelectedVector = []


def get_color_palette_length():
    i = 0
    len = 0
    paletteLength = COLOR_PALETTE.__len__()
    while i < paletteLength:
        len += COLOR_PALETTE[i]['children'].__len__()
        i += 1
    return len


global colorPaletteLength
colorPaletteLength = get_color_palette_length()


def image_to_vec(filename: str) -> list[int]:
    vector = []
    for image in split(filename):
        vector += child_image_to_vec(image)

    return vector


def child_image_to_vec(img: Mat) -> list[int]:
    vector = [0] * colorPaletteLength

    for subimg in img:
        for vec in subimg:
            vector[match_image_color_with_palette_color(vec)['name']] += 1
    return vector


def match_image_color_with_palette_color(vector: list[int]) -> any:
    # Convert RGB color space to HLS color space.
    h, l, s = colorsys.rgb_to_hls(vector[0]/255, vector[1]/255, vector[2]/255)
    # Colorsys returns values between 0 and 1, so we multiply Hue component by 360 to use the whole spectrum.
    h = h*360

    # Match against blacks palette
    if ((l <= 0.07 or (s <= 0.03)) and ((vector[0] <= 15 and vector[1] <= 15 and vector[2] <= 15))):
        return search_color_in_palette(vector, Color.BLACK.value)

    # Match against grays palette
    if (((l > 0.07 and l <= 0.85) and (s >= 0.03 and s < 0.08)) and
        ((vector[0] < 190 and vector[1] < 190 and vector[2] < 190)) and
            ((vector[0] > 15 and vector[1] > 15 and vector[2] > 15))):
        return search_color_in_palette(vector, Color.GREY.value)

    # Match against whites palette
    if ((l > 0.78 or s >= 0.08) and ((vector[0] >= 190 and vector[1] >= 190 and vector[2] >= 190))):
        return search_color_in_palette(vector, Color.WHITE.value)

    # Match against reds palette
    if ((h < 67.5) or (h > 300)):
        return search_color_in_palette(vector, Color.RED.value)

    # Match against greens palette
    if ((h >= 67.5 and h <= 175)):
        return search_color_in_palette(vector, Color.GREEN.value)

    # Match against blues palette
    if ((h > 175 and h <= 300)):
        return search_color_in_palette(vector, Color.BLUE.value)


def save_last_palette_color_for_color(paletteColor: dict):
    global lastPaletteColor
    lastPaletteColor = paletteColor


def save_last_selected_vector(vector: list[int]):
    global lastSelectedVector
    lastSelectedVector = vector


def search_color_in_palette(vector: list[int], matching_color: Literal) -> any:
    if (lastPaletteColor != {} and lastSelectedVector != []) and ((vector == lastSelectedVector).all()):
        return lastPaletteColor

    distanceResult = 0
    colorResult = {}

    # Get the palette color dictionary that matches with the provided color.
    paletteColor = [
        x for x in COLOR_PALETTE if x['values'] == matching_color][0]

    for color in paletteColor['children']:
        p_distance = distance(vector, color['values'])

        if (p_distance == 0):
            return color

        if p_distance < distanceResult or colorResult == {}:
            distanceResult = p_distance
            colorResult = color

    save_last_palette_color_for_color(colorResult)
    save_last_selected_vector(vector)
    return colorResult


def distance(element_1: list[int], element_2: list[int]) -> float:
    result = 0
    result += np.linalg.norm(element_1 - element_2)

    return result


def map_image_to_palette_color(filename: str):
    img = cv2.imread(filename)
    cv2.imshow('original_'+filename, img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    mapped_image = np.zeros(img.shape, dtype=np.uint8)
    # start = time.time()

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
    # end = time.time()
    # print(end - start)
    cv2.waitKey(0)


def split(filename: str) -> list[Mat]:
    list = []
    img = cv2.imread(filename)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # img = center_crop(img, [DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT])
    img = image_resize(img, DEFAULT_IMAGE_WIDTH, DEFAULT_IMAGE_HEIGHT)

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


def image_resize(image, width=None, height=None, inter=cv2.INTER_AREA) -> Mat:
    return cv2.resize(image, (width, height), interpolation=inter)


def center_crop(img, dim) -> Mat:
    width, height = img.shape[1], img.shape[0]

    crop_width = dim[0] if dim[0] < img.shape[1] else img.shape[1]
    crop_height = dim[1] if dim[1] < img.shape[0] else img.shape[0]

    mid_x, mid_y = int(width/2), int(height/2)
    cw2, ch2 = int(crop_width/2), int(crop_height/2)
    return img[mid_y-ch2:mid_y+ch2, mid_x-cw2:mid_x+cw2]

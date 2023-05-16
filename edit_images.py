# import argparse
import glob
import random
from enum import Enum
from PIL import Image, ImageEnhance


class Operations(Enum):
    ROTATE = 1
    BRIGHTEN = 2
    SATURATE = 3


def rotate_image(img: Image) -> Image:
    return img.rotate(randomize_int_range(grade))


def brighten_image(img: Image) -> Image:
    return image_enhancer(img, Operations.BRIGHTEN.value).enhance(randomize_float_range(brightness))


def saturate_image(img: Image) -> Image:
    return image_enhancer(img, Operations.SATURATE.value).enhance(randomize_float_range(saturation))


def randomize_int_range(num: int) -> int:
    return random.randint(1, num)


def randomize_float_range(num: float) -> float:
    return random.uniform(1, num)


def image_enhancer(img: Image, operation: Operations) -> any:
    if (operation == Operations.BRIGHTEN.value):
        return ImageEnhance.Brightness(img)

    if (operation == Operations.SATURATE.value):
        return ImageEnhance.Color(img)


# ap = argparse.ArgumentParser()
# ap.add_argument(
#     "-d",
#     "--directory",
#     required=True,
#     help="Path to the directory of images")

# ap.add_argument(
#     "-n",
#     "--number",
#     required=True,
#     help="Number of images")

# ap.add_argument(
#     "-o",
#     "--output",
#     required=True,
#     help="Objecto")


# args = vars(ap.parse_args())

grade = 15
brightness = 1.5
saturation = 1.5
lightness = 1.5


def edited_images(directory: str, output: str, number: int):
    for imagePath in random.choices(glob.glob(directory + "/*.png"), k=int(number)):
        filename = imagePath[imagePath.rfind("\\") + 1:]
        img = Image.open(imagePath)
        operation = randomize_int_range(len(Operations))

        if operation == Operations.ROTATE.value:
            img = rotate_image(img)

        if operation == Operations.BRIGHTEN.value:
            img = brighten_image(img)

        if operation == Operations.SATURATE.value:
            img = saturate_image(img)

        img.save(f'{output}/{filename}')

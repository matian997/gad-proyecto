import argparse
import glob
import random
from PIL import Image, ImageEnhance


def rotateImage(img: Image) -> Image:
    return img.rotate(my_random(grade))


def brightnessImage(img: Image) -> Image:
    return enhancerImage(img).enhance(my_random(brightness))


def my_random(num: int) -> int:
    return random.randrange(num)


def enhancerImage(img: Image) -> ImageEnhance.Brightness:
    return ImageEnhance.Brightness(img)


ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--directory",
    required=True,
    help="Path to the directory of images")

ap.add_argument(
    "-n",
    "--number",
    required=True,
    help="Number of images")

ap.add_argument(
    "-o",
    "--output",
    required=True,
    help="Objecto")


args = vars(ap.parse_args())

grade = 15
brightness = 1.2
saturation = 0.3
lightness = 0.15
operations = 2

for imagePath in random.choices(glob.glob(args["directory"] + "/*.png"), k=args["number"]):
    filename = imagePath[imagePath.rfind("\\") + 1:]
    img = Image.open(imagePath)

    operation = my_random(operations)

    if operation == 1:
        img = rotateImage(img)

    if operation == 2:
        img = brightnessImage(img)

    img.save(args["output"]+filename)

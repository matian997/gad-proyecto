import argparse
from img_to_vec import map_image_to_palette_color


ap = argparse.ArgumentParser()
ap.add_argument(
    "-i",
    "--image",
    required=True,
    help="Path to the directory of images")

args = vars(ap.parse_args())

map_image_to_palette_color(args["image"])

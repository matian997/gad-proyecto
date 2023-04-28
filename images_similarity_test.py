import argparse
import glob
from PIL import Image
from images_similarity_search import images_similarity_search

ap = argparse.ArgumentParser()

ap.add_argument(
    "-d",
    "--directory",
    required=True,
    help="Path to the directory of images")

ap.add_argument(
    "-r",
    "--radius",
    required=True,
    help="Radius")

args = vars(ap.parse_args())

i = 0
length = glob.glob(args["directory"] + "/*.png").__len__()

for imagePath in glob.glob(args["directory"] + "/*.png"):
    filename = imagePath[imagePath.rfind("/") + 1:]
    original_filename = imagePath[imagePath.rfind("_") + 1:]
    results = images_similarity_search(filename, args["radius"])

    if any(map(lambda x: x[0][x[0].rfind("/")+1:] == original_filename, results)):
        i += 1

print('Success: ', i)
print('Total: ', length)
print('Result: ', i/length)

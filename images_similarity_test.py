import argparse
import glob
import time
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
start = time.time()
for imagePath in glob.glob(args["directory"] + "/*.png"):
    filename = filename = imagePath[imagePath.rfind("\\") + 1:]
    results = images_similarity_search(imagePath, args["radius"])

    list_result = list(
        map(lambda x: x[0][x[0].rfind("\\")+1:], results))

    if any(map(lambda x: x[0][x[0].rfind("\\")+1:] == filename, results)) and list_result.index(filename) < 1:
        i += 1

end = time.time()
print(end - start)
print('Success: ', i)
print('Total: ', length)
print('Result: ', i/length)

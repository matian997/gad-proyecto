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

# Initialize 10 positions since is the limit as per the database.
similarity_search_positions_list = [0] * 10

for imagePath in glob.glob(args["directory"] + "/*.png"):
    imageFileName = imagePath[imagePath.rfind("\\") + 1:]

    results = images_similarity_search(imagePath, args["radius"])
    list_result = list(map(lambda x: x[0][x[0].rfind("\\")+1:], results))

    if any(map(lambda x: x[0][x[0].rfind("\\")+1:] == imageFileName, results)):
        similarity_search_positions_list[list_result.index(imageFileName)] += 1
        i += 1

print('Success: ', i)
print('Total: ', length)
print('Result: ', i/length)
queryResults = open("imageSimilarityQueryResults.txt", 'w+')
queryResults.write(f'Positions: [ {" ".join(str(x) for x in similarity_search_positions_list)} ] \n')

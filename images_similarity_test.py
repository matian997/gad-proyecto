import argparse
import glob
import os
import shutil

from edit_images import edited_images
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

ap.add_argument(
    "-i",
    "--iterations",
    required=True,
    help="Iterations")


args = vars(ap.parse_args())
queryResults = open("imageSimilarityQueryResults.txt", 'w+')
for j in range(int(args["iterations"])):
    print("Jiteration: ", j)
    if os.path.exists(args["output"]):
        shutil.rmtree(args["output"])

    os.makedirs(args["output"])

    edited_images(args["directory"], args["output"], args["number"])

    i = 0
    imagesPath = glob.glob(args["output"] + "/*.png")
    length = imagesPath.__len__()

    # Initialize 10 positions since is the limit as per the database.
    similarity_search_positions_list = [0] * 10

    for imagePath in imagesPath:
        imageFileName = imagePath[imagePath.rfind("\\") + 1:]
        results = images_similarity_search(imagePath, args["radius"])
        list_result = list(map(lambda x: x[0][x[0].rfind("\\")+1:], results))

        if any(map(lambda x: x[0][x[0].rfind("\\")+1:] == imageFileName, results)):
            index = list_result.index(imageFileName)
            similarity_search_positions_list[index] += 1

            if index < 5:
                i += 1

    queryResults.write(
        f'Positions: [ {" ".join(str(x) for x in similarity_search_positions_list)} ], Success: {i}, Total: {length}, Result: {i/length} \n')

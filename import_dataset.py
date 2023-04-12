import argparse
import glob
from PIL import Image
from postgress_connection import getConnection
from img_to_vec import imageToVec

ap = argparse.ArgumentParser()
ap.add_argument(
    "-d",
    "--dataset",
    required=True,
    help="Path to the directory of images")

args = vars(ap.parse_args())
connection = getConnection()
cursor = connection.cursor()

for imagePath in glob.glob(args["dataset"] + "/*.png"):
    filename = imagePath[imagePath.rfind("/") + 1:]
    img = Image.open(filename)
    vec = imageToVec(img)
    print('------ Importing Image: ', filename)
    print(vec)

    cursor.execute(
        """
            INSERT INTO IMAGES(NAME, HISTOGRAM) VALUES(%s, %s)
            """,
        (filename, vec))
    connection.commit()

cursor.close()
connection.close()

print("------ Completed")

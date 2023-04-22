import argparse
import glob
from postgress_connection import getConnection
from img_to_vec import image_to_vec

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
    vec = image_to_vec(filename)

    print('------ Importing Image: ', filename)

    cursor.execute(
        """
            INSERT INTO IMAGES_DATASET(NAME, COLOR_HISTOGRAM_VECTOR) VALUES(%s, %s)
            """,
        (filename, vec))
    connection.commit()

cursor.close()
connection.close()

print("------ Completed")

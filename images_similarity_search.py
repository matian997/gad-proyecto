from postgress_connection import getConnection
from img_to_vec import image_to_vec
import ipyplot


def images_similarity_search(filename: str, radius: int) -> None:
    connection = getConnection()
    cursor = connection.cursor()
    vec = image_to_vec(filename)

    cursor.execute(
        "SELECT IMAGE_NAME, IMAGE_DISTANCE FROM IMAGES_SIMILARITY_SEARCH(%s,%s)", (vec, radius))

    results = cursor.fetchall()
    connection.close()

    images = list(map(lambda x: x[0], results))
    labels = list(map(lambda x: 'Distance: ' + str(x[1]), results))

    ipyplot.plot_images(images, labels, img_width=200)

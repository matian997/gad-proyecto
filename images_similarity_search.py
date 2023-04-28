from postgress_connection import getConnection
from img_to_vec import image_to_vec


def images_similarity_search(filename: str, radius: int) -> list[tuple[any, ...]]:
    connection = getConnection()
    cursor = connection.cursor()
    vec = image_to_vec(filename)

    cursor.execute(
        "SELECT IMAGE_NAME, IMAGE_DISTANCE FROM IMAGES_SIMILARITY_SEARCH(%s,%s)", (vec, radius))

    results = cursor.fetchall()
    connection.close()

    return results

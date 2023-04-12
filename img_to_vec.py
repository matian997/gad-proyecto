from img2vec_pytorch import Img2Vec

img2vec = Img2Vec()


def imageToVec(img):
    return img2vec.get_vec(img, tensor=False)

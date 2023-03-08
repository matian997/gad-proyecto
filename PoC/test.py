import math
from psycopg2 import connect
import cv2
import argparse

ap = argparse.ArgumentParser()
ap.add_argument("-n", "--name", required=True, help="Image Path")
ap.add_argument("-sn", "--sname", required=True, help="Image Path")

args = vars(ap.parse_args())
# channels = [0, 1, 2]
# histSize = [256, 256, 256]
# ranges = [0, 256, 0, 256, 0, 256]

channels = [0]
histSize = [256]
ranges = [0, 256]

hist = cv2.calcHist(
    [cv2.imread(args["name"])],
    channels,
    None,
    histSize,
    ranges)

hist2 = cv2.calcHist(
    [cv2.imread(args["sname"])],
    channels,
    None,
    histSize,
    ranges)

# compute euclidean distance
sum = 0
for i in range(0, 256):
    sum = sum + (hist[i][0]-hist2[i][0])**2
dist = math.sqrt(sum)
print('euclidean distance:', dist)

# above is equivalent to cv2.norm()
dist2 = cv2.norm(hist, hist2, normType=cv2.NORM_L2)
print('euclidean distance2:', dist2)

conn = connect(
    dbname="test_db",
    user="root",
    host="localhost",
    password="root"
)

cursor = conn.cursor()
cursor.callproc(
    "distance",
    (hist.flatten().tolist(),
     hist2.flatten().tolist()),
    512)

result = cursor.fetchall()

print('distance: ', result)

print('compareHist: ', cv2.compareHist(hist, hist2, cv2.HISTCMP_CORREL))

# dist = norm(
#     cv2.calcHist([image], channels, None, histSize, ranges),
#     cv2.calcHist([image2], channels, None, histSize, ranges))

# print(dist)


cursor.close()
conn.close()

# plt.hist(hist, bins=10)
# plt.show()

# print(hist)
# print(hist.__len__())

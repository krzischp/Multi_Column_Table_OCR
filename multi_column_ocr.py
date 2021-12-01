# import the necessary packages
from sklearn.cluster import AgglomerativeClustering
from pytesseract import Output
from tabulate import tabulate
import pandas as pd
import numpy as np
import pytesseract
import argparse
import imutils
import cv2

# construct the argument parser and parse the arguments
ap = argparse.ArgumentParser()

# • --image: The path to the input image containing the table, spreadsheet, etc. that we
# want to detect and OCR
# • --output: Path to the output CSV file that will contain the column data we extracted
# • --min-conf: Used to filter out weak text detections
# • --dist-thresh: Distance threshold cutoff (in pixels) for when applying HAC; you may
# need to tune this value for your images and datasets
# • --min-size: Minimum number of data points in a cluster for it to be considered a
# column

ap.add_argument("-i", "--image", required=True,
                help="path to input image to be OCR'd")
ap.add_argument("-o", "--output", required=True,
                help="path to output CSV file")
ap.add_argument("-c", "--min-conf", type=int, default=0,
                help="minimum confidence value to filter weak text detection")
# Setting your --dist-thresh properly is paramount to OCR’ing multi-column data, be
# sure to experiment with different values.
ap.add_argument("-d", "--dist-thresh", type=float, default=25.0,
                help="distance threshold cutoff for clustering")
# The --min-size command line argument is also important. At each iteration of our
# clustering algorithm, HAC will examine two clusters, each of which could contain multiple data
# points or just a single data point. If the distance between the two clusters is less than the
# --dist-thresh, HAC will merge them.
ap.add_argument("-s", "--min-size", type=int, default=2,
                help="minimum cluster size (i.e., # of entries in column)")
# I added this parameter so we can close gaps in between characters effectively
ap.add_argument("-k", "--kernel-height", type=int, default=15,
                help="kernel height")
args = vars(ap.parse_args())

# set a seed for our random number generator
np.random.seed(42)
# load the input image and convert it to grayscale
image = cv2.imread(args["image"])
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Our next code block detects large blocks of text in our image, taking a similar process to our
# chapter on OCR’ing passports in the “Intro to OCR” Bundle:
# initialize a rectangular kernel that is ~5x wider than it is tall,
# then smooth the image using a 3x3 Gaussian blur and then apply a
# blackhat morphological operator to find dark regions on a light
# background
# kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, 11))
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (51, args["kernel_height"]))

gray = cv2.GaussianBlur(gray, (3, 3), 0)
blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

# compute the Scharr gradient of the blackhat image and scale the
# result into the range [0, 255]
grad = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
grad = np.absolute(grad)
(minVal, maxVal) = (np.min(grad), np.max(grad))
grad = (grad - minVal) / (maxVal - minVal)
grad = (grad * 255).astype("uint8")

# apply a closing operation using the rectangular kernel to close
# gaps in between characters, apply Otsu's thresholding method, and
# finally a dilation operation to enlarge foreground regions
grad = cv2.morphologyEx(grad, cv2.MORPH_CLOSE, kernel)
thresh = cv2.threshold(grad, 0, 255,
                       cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
thresh = cv2.dilate(thresh, None, iterations=3)
cv2.imshow("Thresh", thresh)

# find contours in the thresholded image and grab the largest one,
# which we will assume is the stats table
cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,
                        cv2.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
tableCnt = max(cnts, key=cv2.contourArea)

# compute the bounding box coordinates of the stats table and extract
# the table from the input image

(x, y, w, h) = cv2.boundingRect(tableCnt)
table = image[y:y + h, x:x + w]

# show the original input image and extracted table to our screen
cv2.imshow("Input", image)
cv2.imshow("Table", table)

# Now that we have our statistics table, let’s OCR it:
# set the PSM mode to detect sparse text, and then localize text in
# the table
options = "--psm 6"
results = pytesseract.image_to_data(
    cv2.cvtColor(table, cv2.COLOR_BGR2RGB),
    config=options,
    output_type=Output.DICT)

# initialize a list to store the (x, y)-coordinates of the detected
# text along with the OCR'd text itself
coords = []
ocrText = []

# Let’s move on to looping over each of our text detections:
# loop over each of the individual text localizations
for i in range(0, len(results["text"])):
    # extract the bounding box coordinates of the text region from
    # the current result
    x = results["left"][i]
    y = results["top"][i]
    w = results["width"][i]
    h = results["height"][i]
    # extract the OCR text itself along with the confidence of the
    # text localization
    text = results["text"][i]
    conf = int(results["conf"][i])
    # filter out weak confidence text localizations
    if conf > args["min_conf"]:
        # update our text bounding box coordinates and OCR'd text,
        # respectively
        coords.append((x, y, w, h))
        ocrText.append(text)

# We can now move on to the clustering phase of the project:
# extract all x-coordinates from the text bounding boxes, setting the
# y-coordinate value to zero

# i. First, to apply HAC, we need a set of input vectors (also called “feature vectors”). Our
# input vectors must be at least 2-d, so we add a trivial dimension containing a value of 0.
# ii. Secondly, we aren’t interested in the y-coordinate value. We only want to cluster on the
# x-coordinate positions. Pieces of text with similar x-coordinates are likely to be part of a
# column in a table.
# Once no two clusters have a distance less than --dist-thresh, we stop the
# clustering processing.

# Also, note that we are using the Manhattan distance function here. Why not some other
# distance function, such as Euclidean?
# While you can (and should) experiment with our distance metrics, Manhattan tends to be an
# appropriate choice here. We want to be very stringent on our requirement that x-coordinates
# lie close together. But again, I suggest you experiment with other distance
xCoords = [(c[0], 0) for c in coords]
# apply hierarchical agglomerative clustering to the coordinates
clustering = AgglomerativeClustering(
    n_clusters=None,
    affinity="manhattan",
    linkage="complete",
    distance_threshold=args["dist_thresh"])
clustering.fit(xCoords)
# initialize our list of sorted clusters
sortedClusters = []


# Now that our clustering is complete, let’s loop over each of the unique clusters:
# loop over all clusters
for l in np.unique(clustering.labels_):
    # extract the indexes for the coordinates belonging to the
    # current cluster
    idxs = np.where(clustering.labels_ == l)[0]
    # verify that the cluster is sufficiently large
    if len(idxs) > args["min_size"]:
        # compute the average x-coordinate value of the cluster and
        # update our clusters list with the current label and the
        # average x-coordinate
        avg = np.average([coords[i][0] for i in idxs])
        sortedClusters.append((l, avg))

# sort the clusters by their average x-coordinate and initialize our
# data frame
sortedClusters.sort(key=lambda x: x[1])
df = pd.DataFrame()

# Let’s now loop over our sorted clusters:
# loop over the clusters again, this time in sorted order
for (l, _) in sortedClusters:
    # extract the indexes for the coordinates belonging to the
    # current cluster
    idxs = np.where(clustering.labels_ == l)[0]
    # extract the y-coordinates from the elements in the current
    # cluster, then sort them from top-to-bottom
    yCoords = [coords[i][1] for i in idxs]
    sortedIdxs = idxs[np.argsort(yCoords)]
    # generate a random color for the cluster
    color = np.random.randint(0, 255, size=(3,), dtype="int")
    color = [int(c) for c in color]

    # Let’s loop over each of the pieces of text in the column now:
    # loop over the sorted indexes
    for i in sortedIdxs:
        # extract the text bounding box coordinates and draw the
        # bounding box surrounding the current element
        (x, y, w, h) = coords[i]
        cv2.rectangle(table, (x, y), (x + w, y + h), color, 2)
    # extract the OCR'd text for the current column, then construct
    # a data frame for the data where the first entry in our column
    # serves as the header
    cols = [ocrText[i].strip() for i in sortedIdxs]
    currentDF = pd.DataFrame({cols[0]: cols[1:]})
    # concatenate *original* data frame with the *current* data
    # frame (we do this to handle columns that may have a varying
    # number of rows)
    df = pd.concat([df, currentDF], axis=1)


# At this point our table OCR process is complete, we just need to save the table to disk:
# replace NaN values with an empty string and then show a nicely
# formatted version of our multi-column OCR'd text
df.fillna("", inplace=True)
print(tabulate(df, headers="keys", tablefmt="psql"))
# write our table to disk as a CSV file
print("[INFO] saving CSV file to disk...")
df.to_csv(args["output"], index=False)
# show the output image after performing multi-column OCR
cv2.imshow("Output", image)
cv2.waitKey(0)

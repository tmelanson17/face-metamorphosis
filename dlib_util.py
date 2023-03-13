# import the necessary packages
import facial_utils
import numpy as np
import cv2
import argparse
import dlib
from facial_regions import get_index_pairs

DEFAULT_SHAPE_PREDICTOR="shape_predictor_68_face_landmarks.dat"


def extract_facial_landmarks(face_image, predictor_file=DEFAULT_SHAPE_PREDICTOR):
    # initialize dlib's face detector (HOG-based) and then create
    # the facial landmark predictor
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor(predictor_file)

    # Convert face to grayscale
    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    # Use the first rectangle detected for the face.
    rect = rects[0]

    # determine the facial landmarks for the face region, then
    # convert the facial landmark (x, y)-coordinates to a NumPy
    # array
    shapes = predictor(gray, rect)
    shapes = facial_utils.shape_to_np(shapes)

    # Return main face rectangle and list of (x, y)-coordinates
    return shapes, rect


def convert_landmarks_to_lines(points, facial_regions):
    return [
        (points[i][1], points[i][0], points[j][1], points[j][0])
        for (i, j) in facial_regions]


if __name__ == "__main__":
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--shape-predictor", required=True,
            help="path to facial landmark predictor")
    ap.add_argument("-i", "--image", required=True,
            help="path to input image")
    args = vars(ap.parse_args())

    # load the input image, resize it, and convert it to grayscale
    image = cv2.imread(args["image"])

    rect, shapes = extract_facial_landmarks(image, args["shape_predictors"])

    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    (x, y, w, h) = facial_utils.rect_to_bb(rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face #{}".format(i + 1), (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the face detections
    for (i, (x,y)) in enumerate(shapes):
            # loop over the (x, y)-coordinates for the facial landmarks
            # and draw them on the image
            cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    # show the output image with the face detections + facial landmarks
    cv2.imshow("Output", image)
    cv2.waitKey(0)

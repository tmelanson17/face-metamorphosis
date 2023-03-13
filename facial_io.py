import cv2
import os
import numpy as np
import argparse

FACE_CASCADE_XML = 'haarcascade_frontalface_alt.xml'


def read_faces(args):
    faces = list()
    for img_name in args["images"]:
        faces.append(cv2.imread(img_name))
    return faces


def parse_names(args):
    names = list()
    for fullpath in args["images"]:
        names.append(
            os.path.splitext(os.path.basename(fullpath))[0]
        )
    return names


def crop_face(frame, dst_shape):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray = cv2.equalizeHist(frame_gray)
    #-- Detect faces
    face_cascade = cv2.CascadeClassifier()
    face_cascade.load(FACE_CASCADE_XML)
    facex, facey, face_width, face_height = face_cascade.detectMultiScale(frame_gray)[0]
    face = np.copy(frame[facey:(facey+face_height), facex:(facex+face_width)])
    return cv2.resize(face, dst_shape[::-1])


# Create a smooth mixture of two images.
def mix_images(image1, image2, alpha):
    mix = (image1 * (1 - alpha) + image2 * alpha).astype(np.uint8)
    # Mute any black regions
    mix[image1 == 0] = image2[image1 == 0]
    mix[image2 == 0] = image1[image2 == 0]
    return mix


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, nargs="+",
                    help="List of images to create interpolations of (will be done for each consecutive face)")
    args = vars(ap.parse_args())
    images = read_faces(args)
    names = parse_names(args)
    try:
        os.makedirs("cropped_faces")
    except FileExistsError:
        pass
    for img, name in zip(images, names):
        face_cropped = crop_face(img, (200, 200))
        cv2.imwrite(os.path.join("cropped_faces", name + ".png"), face_cropped)

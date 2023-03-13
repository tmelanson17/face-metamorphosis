import dlib_util
import metamorphosis
import facial_regions
import facial_io
import argparse
import cv2
import os
import numpy as np


def write_video_directory(video_frames, output_dir, sequence_name, video_name):
    video_dir = os.path.join(output_dir, sequence_name, video_name)
    try:
        os.makedirs(video_dir)
    except FileExistsError:
        pass
    for i, frame in enumerate(video_frames):
        cv2.imwrite(
            os.path.join(video_dir, "{0:s}_{1:04d}.png".format(video_name, i)),
            frame
        )


def create_facial_interpolation(face_1, face_2, n_frames):
    points1, _ = dlib_util.extract_facial_landmarks(face_1)

    feat1 = dlib_util.convert_landmarks_to_lines(points1, facial_regions.get_index_pairs())
    points2, _ = dlib_util.extract_facial_landmarks(face_2)
    feat2 = dlib_util.convert_landmarks_to_lines(points2, facial_regions.get_index_pairs())

    # Determine f ( dest coordinates) => source coordinates
    X_src = metamorphosis.compute_weighted_source_coordinates(face_2.shape[:2], feat1, feat2)

    # Determine f ( src coordinates) => dest coordinates
    X_dst = metamorphosis.compute_weighted_source_coordinates(face_1.shape[:2], feat2, feat1)

    # Determine pixel coordinates along each intermediate frame
    interpolation_src = metamorphosis.create_interpolation(face_1.shape[:2], X_src, n_frames=n_frames)
    interpolation_dst = metamorphosis.create_interpolation(face_2.shape[:2], X_dst, n_frames=n_frames)

    src_warped = [metamorphosis.warp_source(face_1, X_src, pixel_est_method='bilinear', pad='edge')
                  for X_src in interpolation_src]
    dst_warped = [metamorphosis.warp_source(face_2, X_dst, pixel_est_method='bilinear', pad='edge')
                  for X_dst in interpolation_dst]

    mix_warped = [
                facial_io.mix_images(src_warped[i], dst_warped[n_frames-i], i / n_frames)
                for i in range(n_frames+1)
    ]

    return src_warped, dst_warped, mix_warped


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", required=True, nargs="+",
                    help="List of images to create interpolations of (will be done for each consecutive face)")
    ap.add_argument("--frames", type=int,
                    help="Number of frames of interpolation for the transformation.")
    ap.add_argument("--output-dir", default="output", 
                    help="Output directory of the interpolations.")
    face_shape = (200, 200) # Rows by columns
    args = vars(ap.parse_args())
    images_uncropped = facial_io.read_faces(args)
    images = [facial_io.crop_face(img, face_shape) for img in images_uncropped]
    names = facial_io.parse_names(args)
    output_dir = args["output_dir"]

    for i in range(len(images)):
        ip1 = (i+1) % len(images)
        image_1 = images[i]
        image_2 = images[ip1]
        name_1 = names[i]
        name_2 = names[ip1]

        src, dst, mix = create_facial_interpolation(image_1, image_2, args["frames"])

        sequence_dir = "{}_to_{}".format(name_1, name_2)
        write_video_directory(src, output_dir, sequence_dir, "src")
        write_video_directory(dst, output_dir, sequence_dir, "dst")
        write_video_directory(mix, output_dir, sequence_dir, "mix")


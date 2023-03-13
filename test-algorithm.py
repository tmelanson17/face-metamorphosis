import cv2
import numpy as np
import scipy as sp
import unittest
import os
import math

import metamorphosis
import dlib_util
import facial_regions

DST_FOLDER = os.path.abspath("output")
EXTS = ['.bmp', '.pbm', '.pgm', '.ppm', '.sr', '.ras', '.jpeg', '.jpg',
        '.jpe', '.jp2', '.tiff', '.tif', '.png']


def write_video_directory(name, video_frames):
    video_dir = os.path.join(DST_FOLDER, "images", name)
    try:
        os.makedirs(video_dir)
    except FileExistsError:
        pass
    for i, frame in enumerate(video_frames):
        cv2.imwrite(
            os.path.join(video_dir, "{}_{}.png".format(name, i)),
            frame
        )


class MetamorphosisAlgorithmTest(unittest.TestCase):

    def setUp(self):
        self.orig_image = np.array([
            [0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0]
            ], dtype=np.float)
        self.new_image = np.array([
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 1],
            [0, 0, 0, 0, 1, 0, 1],
            ], dtype=np.float)
        self.img_shape = self.orig_image.shape
        self.dst_shape = self.new_image.shape
        # Source line
        self.P_src = (0, 2)
        self.Q_src = (0, 4)
        # Destination line
        self.P_dst = (0, 6)
        self.Q_dst = (4, 6)

        self.face_1 = cv2.imread("face1.jpeg")
        self.face_2 = cv2.imread("face2.jpeg")
        self.face_2 = cv2.resize(self.face_2, (self.face_1.shape[1], self.face_1.shape[0]))

    # Test that the image finds the correct (u,v) pairs for a given line
    def testLineDistance(self):
        u_expected = np.tile((np.arange(5, dtype=np.float) - 2) / 2, (5, 1))
        v_expected = np.tile(np.arange(5, dtype=np.float)[:, np.newaxis], (1, 5))
        u,v = metamorphosis.compute_uv(self.img_shape, self.P_src, self.Q_src)
        self.assertTrue(np.all(u_expected == u))
        self.assertTrue(np.all(v_expected == v))

    # Test that a 90 degree rotation and doubling in size works
    # Expects that uv computation works.
    def testReprojCoords(self):
        X1_dst = (0, 2)
        X2_dst = (4, 6)
        X1_src_expected = np.array([4, 2])
        X2_src_expected = np.array([0, 4])
        src_coord = metamorphosis.find_reproj_coord(self.dst_shape, self.P_dst, self.Q_dst, self.P_src, self.Q_src)
        self.assertTrue(np.all(src_coord[X1_dst] == X1_src_expected)) 
        self.assertTrue(np.all(src_coord[X2_dst] == X2_src_expected)) 

    # Test that the reprojection (with floored source indices) behaves as expected
    def testReproj(self):
        src_coord = metamorphosis.find_reproj_coord(self.dst_shape, self.P_dst, self.Q_dst, self.P_src, self.Q_src)
        dst = metamorphosis.warp_source(self.orig_image, src_coord, pixel_est_method='floor')
        expected_image = np.atleast_3d(self.new_image)
        self.assertEqual(len(expected_image.shape), len(dst.shape))
        self.assertEqual(expected_image.shape[0], dst.shape[0])
        self.assertEqual(expected_image.shape[1], dst.shape[1])
        for i in range(expected_image.shape[0]):
            for j in range(expected_image.shape[1]):
                self.assertEqual(expected_image[i,j], dst[i,j])

    # Test that the coordinates are extended out from the original image value
    # by doing the projection of the zoomed in image to the zoomed out image
    def testDownsizedDstImage(self):
        src_coord = metamorphosis.find_reproj_coord(self.img_shape, self.P_src, self.Q_src, self.P_dst, self.Q_dst)
        dst = metamorphosis.warp_source(self.new_image, src_coord, pixel_est_method='floor')
        expected_image = np.atleast_3d(self.orig_image)
        self.assertEqual(len(expected_image.shape), len(dst.shape))
        self.assertEqual(expected_image.shape[0], dst.shape[0])
        self.assertEqual(expected_image.shape[1], dst.shape[1])
        for i in range(expected_image.shape[0]):
            for j in range(expected_image.shape[1]):
                self.assertEqual(expected_image[i,j], dst[i,j])
        self.assertTrue(np.all(expected_image == dst))

    # Sanity check for the distance measurement function
    def testComputeDist(self):
        P = (3,2)
        D = (3,4)
        length = 4
        img_shape = (7, 7)
        expected_output = np.array(
                [
                 [math.sqrt(13),math.sqrt(10),3.,3.,3.,math.sqrt(10),math.sqrt(13)],
                 [math.sqrt(8),math.sqrt(5),2.,2.,2.,math.sqrt(5),math.sqrt(8)],
                 [math.sqrt(5),math.sqrt(2),1.,1.,1.,math.sqrt(2),math.sqrt(5)],
                 [2.,1.,0.,0.,0.,1.,2.],
                 [math.sqrt(5),math.sqrt(2),1.,1.,1.,math.sqrt(2),math.sqrt(5)],
                 [math.sqrt(8),math.sqrt(5),2.,2.,2.,math.sqrt(5),math.sqrt(8)],
                 [math.sqrt(13),math.sqrt(10),3.,3.,3.,math.sqrt(10),math.sqrt(13)]
                 ],dtype=np.float)
        u, v = metamorphosis.compute_uv(img_shape, P, D)
        D = metamorphosis.computeDist(length, u, v)
        for i in range(D.shape[0]):
            for j in range(D.shape[1]):
                self.assertEqual(D[i,j],
                        expected_output[i,j])

    # Loads two face images, with pre-determined features, and displays the resulting warp
    # of the first face onto the second.
    def testWarpImage1SanityCheck(self):
        feat1 = ((73, 93, 73, 126), (74, 150, 74, 184), (182, 109, 182, 239))
        feat2 = ((82, 107, 82, 129), (83, 140, 83, 160), (150, 109, 150, 204))
        X_src = metamorphosis.compute_weighted_source_coordinates(self.face_2.shape[:2], feat1, feat2)
        face_1_warped = metamorphosis.warp_source(self.face_1, X_src)
        cv2.imshow("Non-warped face", self.face_1)
        cv2.imshow("Referenced face", self.face_2)
        cv2.imshow("Warped face", face_1_warped)
        # TODO: Write write functions for each of these images

    def testCollectImageRegions(self):
        points, _ = dlib_util.extract_facial_landmarks(self.face_2)
        self.assertEqual(len(points), 68)
        feat = dlib_util.convert_landmarks_to_lines(points, facial_regions.get_index_pairs())
        output_img = np.copy(self.face_2)
        for (x, y) in points:
            cv2.circle(output_img, (x, y), 1, (255, 0, 0), -1)
        for (x1, y1, x2, y2) in feat:
            output_img = cv2.line(output_img, (y1, x1), (y2, x2), (0, 0, 255), 2)
        cv2.imshow("Face", output_img)
        # TODO: Write write functions for each of these images

    def testFullPipelineSanityCheck(self):
        points1, _ = dlib_util.extract_facial_landmarks(self.face_1)
        feat1 = dlib_util.convert_landmarks_to_lines(points1, facial_regions.get_index_pairs())
        points2, _ = dlib_util.extract_facial_landmarks(self.face_2)
        feat2 = dlib_util.convert_landmarks_to_lines(points2, facial_regions.get_index_pairs())
        X_src = metamorphosis.compute_weighted_source_coordinates(self.face_2.shape[:2], feat1, feat2)
        face_1_warped = metamorphosis.warp_source(self.face_1, X_src)
        cv2.imshow("Non-warped face", self.face_1)
        cv2.imshow("Referenced face", self.face_2)
        cv2.imshow("Warped face", face_1_warped / (2. * 255.) +self.face_2 / (2. * 255.))
        # TODO: Write write functions for each of these images

    def testImageInterpolation(self):
        points1, _ = dlib_util.extract_facial_landmarks(self.face_1)
        feat1 = dlib_util.convert_landmarks_to_lines(points1, facial_regions.get_index_pairs())
        points2, _ = dlib_util.extract_facial_landmarks(self.face_2)
        feat2 = dlib_util.convert_landmarks_to_lines(points2, facial_regions.get_index_pairs())

        # Determine f ( dest coordinates) => source coordinates
        X_src = metamorphosis.compute_weighted_source_coordinates(self.face_2.shape[:2], feat1, feat2)

        # Determine f ( src coordinates) => dest coordinates
        X_dst = metamorphosis.compute_weighted_source_coordinates(self.face_1.shape[:2], feat2, feat1)

        # Determine pixel coordinates along each intermediate frame
        interpolation_src = metamorphosis.create_interpolation(self.face_1.shape[:2], X_src, n_frames=30)
        interpolation_dst = metamorphosis.create_interpolation(self.face_2.shape[:2], X_dst, n_frames=30)

        src_warped = [metamorphosis.warp_source(self.face_1, X_src) for X_src in interpolation_src]
        dst_warped = [metamorphosis.warp_source(self.face_2, X_dst) for X_dst in interpolation_dst]
        write_video_directory("src", src_warped)
        write_video_directory("dst", dst_warped)

        mix_warped = [((src*((30 - i) / (30 * 255.)) + dst*(i / (30 * 255.))) * 255.).astype(np.uint8)
                      for i, (src, dst) in enumerate(zip(src_warped, reversed(dst_warped)))]

        write_video_directory("mix", mix_warped)


if __name__ == '__main__':
    unittest.main()

import numpy as np


# Computes the non normalized weight of a given line when warping a point.
# a : either small (full control over warping) or large (smoother warping)
# p in [0, 1] : each line weighted the same to each line weighted on length.
# b in [0.5, 2] : 0.5 evens weight between lines, 2 accentuates it. 1 does nothing.
def computeWeight(length, dist, a=0.01, p=0, b=1):
    return np.power(np.power(length, p) / (a + dist), b)


# Find the dist parameter used for computeWeight
# param length The length of PQ
# param u The distance along PQ (sqrt(length) is the total distance)
# param v The distance perpendicular to PQ (absolute)
def computeDist(length, u, v):
    D = np.zeros_like(u)
    D[u < 0] = np.sqrt(length*np.square(u[u < 0]) + np.square(v[u < 0])).flatten()
    D[u > 1] = np.sqrt(length*np.square((u[u > 1]-1.0)) + np.square(v[u > 1])).flatten()
    D[np.logical_and(u >= 0, u <= 1)] = v[np.logical_and(u >= 0, u <= 1)]
    return np.abs(D)


'''
The purpose of this function is to create a coordinate grid that, given an input shape, returns a stacked array 
of i and j coordinates.
'''
def createCoordinateGrid(img_shape):
    xi, xj = np.meshgrid(np.arange(img_shape[0]), np.arange(img_shape[1]), indexing='ij')
    return np.dstack([xi, xj])


'''
Because the line segment for PQ, as well as its normal vector and total length, always needed to be found, I decided
to create a helper function for it.
'''
def computeAxesAndNorm(P, Q):
    PQ_vector = np.array(Q) - np.array(P)
    # Rotate forward by 90 degrees (=-90)
    PQ_perp = np.array([PQ_vector[1], -PQ_vector[0]])
    PQ_norm = np.linalg.norm(PQ_vector)
    return PQ_vector, PQ_perp, PQ_norm


'''
Given a the resulting image shape, as well as the line defined by initial point P and end point Q, return u, which
is the distance of X along PQ with respect to the length of PQ; and v, which is the distance of X from the line defined 
by PQ.
'''
def compute_uv(img_shape, P, Q):
    X = createCoordinateGrid(img_shape)
    PQ_vector, PQ_perp, PQ_norm = computeAxesAndNorm(P, Q)
    PX_vector = X - P
    return np.dot(PX_vector, PQ_vector)/np.square(PQ_norm), \
            np.dot(PX_vector, PQ_perp)/PQ_norm


'''
Finds the coordinates used for projecting the source image into the destination reference frame.
param[in] dst_shape The size of the output projection coordinate matrix
param[in] P_dst, Q_dst The initial and end points of the facial line in the destination image.
param[in] P_src, Q_src The initial and end points of the corresponding facial line in the source image.
return A 3D-matrix, that, at each 2D index (i1, j1), specifies an (i2,j2) coordinate such that 
output[i1, j1] = source[i2, j2].
'''
def find_reproj_coord(dst_shape, P_dst, Q_dst, P_src, Q_src):
    u, v = compute_uv(dst_shape, P_dst, Q_dst)
    u = np.atleast_3d(u)
    v = np.atleast_3d(v)
    P_src_broadcast = np.broadcast_to(np.array(P_src), (dst_shape[0], dst_shape[1], 2))
    PQ_src_vector, PQ_src_perp, PQ_src_norm = computeAxesAndNorm(P_src, Q_src)
    PQ_src_vector_broadcast = np.broadcast_to(np.array(PQ_src_vector), (dst_shape[0], dst_shape[1], 2))
    PQ_src_perp_broadcast = np.broadcast_to(np.array(PQ_src_perp), (dst_shape[0], dst_shape[1], 2))

    # Xp = Pp + u*(Qp-Pp) + v*Perp(Qp-Pp)/norm(Qp-Pp)
    X_prime = P_src_broadcast + u*PQ_src_vector_broadcast + v*PQ_src_perp_broadcast/PQ_src_norm
    return X_prime


'''
Given a source image and a set of coordinates to be used to populate the destination image, populate the destination 
image with the warped source image.
param[in] pixel_est_method How to extimate a pixel value when the exact value doesn't exist. Can be 'floor', where the
indices are truncated to be integer indices, or 'bilinear', where the pixel is computed as a blend of the four 
neighboring indices.
param[in] pad How to pad the image. Right now supports 'zero' (pad with zeros), but could also pad with edge values.
return Source image warped to the destination reference frame.
'''
def warp_source(src, dst_coord, pixel_est_method='floor', pad='zero'):
    src = np.atleast_3d(src)
    if pad == 'zero' or pad == 'edge':
        pad_left = abs(int(np.minimum(0, np.floor(np.min(dst_coord[:,:,0])))))
        pad_right = int(np.maximum(0, np.max(dst_coord[:,:,0] - src.shape[0]+2)))
        pad_up = abs(int(np.minimum(0, np.floor(np.min(dst_coord[:,:,1])))))
        pad_down = int(np.maximum(0, np.max(dst_coord[:,:,1] - src.shape[1]+2)))
        # Set the value so it can be used with np pad
        if pad == 'zero':
            pad = 'constant'
        src = np.pad(src, ((pad_left, pad_right), (pad_up, pad_down), (0,0)), mode=pad)
        dst_coord[:,:,0] += pad_left
        dst_coord[:,:,1] += pad_up
    else:
        raise IOException('Pad format not recognized')

    if len(src.shape) < 3:
        dst = np.zeros(dst_coord.shape[:2], dtype=src.dtype)
    else:
        dst = np.zeros(dst_coord.shape[:2] + (src.shape[2],), dtype=src.dtype)

    # TODO: Optimize this to not use a for loop
    if pixel_est_method == 'floor':
        dst_coord_est = np.floor(dst_coord).astype(np.uint16)
        for i in range(dst.shape[0]):
            for j in range(dst.shape[1]):
                try:
                    dst[i, j] = src[tuple(dst_coord_est[i, j])]
                except IndexError as e:
                    print("At index [{}, {}]".format(i, j))
                    raise e
    elif pixel_est_method == 'bilinear':
        dst_coord_topleft = np.floor(dst_coord).astype(np.uint16)
        dst_coord_bottomright = np.ceil(dst_coord).astype(np.uint16)
        dst_coord_topright = np.dstack([dst_coord_topleft[:, :, 0], dst_coord_bottomright[:, :, 1]])
        dst_coord_bottomleft = np.dstack([dst_coord_bottomright[:, :, 0], dst_coord_topleft[:, :, 1]])
        blend_alpha = dst_coord - dst_coord_topleft
        for i in range(dst.shape[0]):
            for j in range(dst.shape[1]):
                try:
                    Q11 = src[tuple(dst_coord_topleft[i, j])]
                    Q12 = src[tuple(dst_coord_topright[i, j])]
                    Q21 = src[tuple(dst_coord_bottomleft[i, j])]
                    Q22 = src[tuple(dst_coord_bottomright[i, j])]
                    alphax = blend_alpha[i, j, 0]
                    alphay = blend_alpha[i, j, 1]
                    pixel = (1-alphay)*(1-alphax)*Q11 + (1-alphay)*alphax*Q21 + \
                            (alphay)*(1-alphax)*Q12 + alphay*alphax*Q22
                    dst[i, j] = pixel
                except IndexError as e:
                    print("At index [{}, {}]".format(i, j))
                    raise e
    else:
        raise IOException('Pixel estimation method not recognized')
    
    return dst


'''
Computes the weighted average of the coordinate offset from two sets of (P,Q) pairs, then uses it to create the
reference coordinate matrix.
param[in] dst_shape The shape of the resulting output.
param[in] face1 The set of (P,Q) pairs corresponding to the first face.
param[in] face2 The set of (P,Q) pairs corresponding to the second face.
return[in] The reference coordinate matrix describing what source image pixels correspond to each destination image location.
'''
def compute_weighted_source_coordinates(dst_shape, face1, face2):
    D_sum = np.zeros(dst_shape + (2,))
    weight_sum = np.zeros(dst_shape)
    X = createCoordinateGrid(dst_shape)
    for feat1, feat2 in zip(face1, face2):
        P_src = np.array(feat1[:2])
        Q_src = np.array(feat1[2:])
        P_dst = np.array(feat2[:2])
        Q_dst = np.array(feat2[2:])

        length = np.linalg.norm(Q_dst - P_dst)
        u, v = compute_uv(dst_shape, P_dst, Q_dst)
        dist = computeDist(length, u, v)
        weight = computeWeight(length, dist, a=0.0001, p=1, b=2)
        X_src = find_reproj_coord(dst_shape, P_dst, Q_dst, P_src, Q_src)
        
        D = X_src.astype(np.float) - X.astype(np.float)
        D_sum += np.atleast_3d(weight) * D
        weight_sum += weight
    return X + D_sum / np.atleast_3d(weight_sum)


'''
From a function mapping destination to source coordinates, creates an interpolation of size 1 / n_frames of the
coordinate distance.
'''
def create_interpolation(dst_shape, X_src, n_frames):
    X = createCoordinateGrid(dst_shape)
    frames = list()
    for i in range(n_frames+1):
        alpha = i / n_frames
        D = X_src - X
        frames.append(X_src - (1-alpha)*D)
    return frames

import numpy as np

def get_extrinsics_param(H, intrinsics_param):
    extrinsics_param = []

    inv_intrinsics_param = np.linalg.inv(intrinsics_param)

    for i in range(len(H)):
        h0 = (H[i].reshape(3, 3))[:, 0]
        h1 = (H[i].reshape(3, 3))[:, 1]
        h2 = (H[i].reshape(3, 3))[:, 2]

        scale_factor1 = 1.0 / np.linalg.norm(np.dot(intrinsics_param, h0))
        scale_factor2 = 1.0 / np.linalg.norm(np.dot(intrinsics_param, h1))

        #r0 = scale_factor * np.dot(inv_intrinsics_param, h0)
        #r1 = scale_factor * np.dot(inv_intrinsics_param, h1)
        #t = scale_factor * np.dot(inv_intrinsics_param, h2)

        r0 = scale_factor1 * np.dot(intrinsics_param, h0)
        r1 = scale_factor2 * np.dot(intrinsics_param, h1)
        t = scale_factor1 * np.dot(intrinsics_param, h2)

        r2 = np.cross(r0, r1) / np.linalg.norm(np.cross(r0, r1))

        r3 = np.dot(r0, r1)

        R = np.array([r0, r1, r2, t]).transpose()

        R0 = R[:,0:3]

        U, S, VT = np.linalg.svd((np.array(R0, dtype='float')))

        R[:,0:3] = np.dot(U, VT)

        extrinsics_param.append(R)

    return np.array(extrinsics_param)
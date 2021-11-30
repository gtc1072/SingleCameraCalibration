import cv2 as cv
import numpy as np
import os
from homography import get_homography
from intrinsics import get_intrinsics_param
from extrinsics import get_extrinsics_param
from distortion import get_distortion
from refine_all import refine_all_param

if __name__ == '__main__':
    #print('PyCharm'）
    file_dir = r'./chess'
    pic_name = os.listdir(file_dir)
    cross_corners = [8, 6]
    real_coor = np.zeros((cross_corners[0]*cross_corners[1],3), np.float32)
    real_coor[:,:2]=np.mgrid[0:8,0:6].T.reshape(-1,2)
    real_coor = real_coor * 25
    pic_points=[]
    real_points=[]
    real_points_x_y=[]
    for pic in pic_name:
        pic_path = os.path.join(file_dir, pic)
        pic_data = cv.imread(pic_path, cv.IMREAD_GRAYSCALE)

        succ, pic_coor = cv.findChessboardCorners(pic_data, (cross_corners[0],cross_corners[1]), None)

        if succ:
            pic_coor = pic_coor.reshape(-1,2)
            pic_points.append(pic_coor)
            real_points.append(real_coor)
            real_points_x_y.append(real_coor[:,:2])

    H=get_homography(pic_points, real_points)
    intrinsics_param = get_intrinsics_param(H)
    extrinsics_param = get_extrinsics_param(H, intrinsics_param)
    k=get_distortion(intrinsics_param,extrinsics_param,pic_points,real_points_x_y)
    [new_intrinsics_param, new_k, new_extrinsics_param] = refine_all_param(intrinsics_param,
                                                                           k,
                                                                           extrinsics_param,
                                                                           real_points,
                                                                           pic_points)
    print(H)
    print("\n")
    print(intrinsics_param)
    print("\n")
    print(extrinsics_param)
    print("\n")
    print(k)
    print("\n")
    print("intrinsics_param:\n", new_intrinsics_param)
    print("\ndistortion:\n", new_k)
    print("\nextrinsics_param:\n", new_extrinsics_param)

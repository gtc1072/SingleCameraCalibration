import numpy as np

def get_distortion(intrinsic_param, extrinsic_param, pic_coor, real_coor):
    D=[]
    d=[]
    for i in range(len(pic_coor)):
        for j in range(len(pic_coor[i])):
            single_coor = np.array([(real_coor[i])[j,0],(real_coor[i])[j,1],0,1])
            u=np.dot(np.dot(intrinsic_param, extrinsic_param[i]), single_coor)
            [u_estim, v_estim] = [u[0]/u[2],u[1]/u[2]]

            coor_norm = np.dot(extrinsic_param[i], single_coor)
            coor_norm /= coor_norm[-1]

            r=np.linalg.norm(coor_norm)

            D.append(np.array([(u_estim - intrinsic_param[0, 2]) * r ** 2, (u_estim - intrinsic_param[0, 2]) * r ** 4]))
            D.append(np.array([(u_estim - intrinsic_param[1, 2]) * r ** 2, (u_estim - intrinsic_param[1, 2]) * r ** 4]))

            d.append(pic_coor[i][j,0]-u_estim)
            d.append(pic_coor[i][j,1]-v_estim)

    D=np.array(D)
    temp=np.dot(np.linalg.inv(np.dot(D.T,D)),D.T)
    k=np.dot(temp, d)

    return k
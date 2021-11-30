import numpy as np

def create_v(p,q,H):
    H=H.reshape(3,3)
    return np.array([
        H[0, p] * H[0, q],
        H[0, p] * H[1, q] + H[1,p] * H[0,q],
        H[1, p] * H[1, q],
        H[2, p] * H[0, q] + H[0, p] * H[2, q],
        H[2, p] * H[1, q] + H[1, p] * H[2, q],
        H[2, p] * H[2, q]
    ])

def get_intrinsics_param(H):
    V=[]
    for i in range(len(H)):
        V.append(np.array([create_v(0, 1, H[i])]))
        V.append(np.array([create_v(0, 0, H[i]) - create_v(1, 1, H[i])]))

    U, S, VT = np.linalg.svd((np.array(V, dtype='float')).reshape((-1, 6)))
    b=VT[-1]

    w=b[0]*b[2]*b[5]-b[1]*b[1]*b[5]-b[0]*b[4]*b[4]+2.0*b[1]*b[3]*b[4]-b[2]*b[3]*b[3]
    d=b[0]*b[2]-b[1]*b[1]

    alpha=np.sqrt(w/(d*b[0]))
    beta = np.sqrt(w/d**2*b[0])
    gamma = np.sqrt(w/(d**2*b[0]))*b[1]
    uc=(b[1]*b[4]-b[2]*b[3])/d
    vc=(b[1]*b[3]-b[0]*b[4])/d

    return np.array([
        [alpha, gamma, uc],
        [0,     beta,  vc],
        [0,     0,     1]
    ])

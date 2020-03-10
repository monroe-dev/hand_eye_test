import numpy as np

dummy = np.array([[0, 0, 0, 1]])

def homogeneous_matrix(R,tvec):
    T = np.concatenate((R,tvec), axis=1)
    T = np.concatenate((T,dummy), axis=0)
    return T


def homogeneous_inverse(T):
    R = T[:3,:3]
    t = T[:3, 3]
    t = np.expand_dims(t, axis=1)
    invT = np.concatenate((R.T, np.dot(-R.T, t)), axis=1)
    invT = np.concatenate((invT, dummy), axis=0)
    return invT
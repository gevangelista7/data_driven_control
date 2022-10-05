import numpy as np
import pysindy as ps
import matplotlib.pyplot as plt
from func_library_simple import custom_library

from scipy.io import loadmat
import warnings
warnings.simplefilter("ignore", category=UserWarning)
warnings.simplefilter("ignore", category=FutureWarning)

if __name__ == '__main__':

    # V = loadmat('data/V.mat')['V']
    # q = loadmat('data/q.mat')['q']
    # t = loadmat('data/t.mat')['t'].squeeze()
    # theta = loadmat('data/theta.mat')['theta']
    # params = loadmat('data/params.mat')['params']
    # vx = loadmat('data/vx.mat')['vx']
    # vz = loadmat('data/vz.mat')['vz']
    # x = loadmat('data/x.mat')['x']
    # z = loadmat('data/z.mat')['z']
    # delta = loadmat('data/delta.mat')['delta']

    mat_file = loadmat('data/exp_mach1d5_alt5000m_v3.mat')
    V = mat_file['V']
    x = mat_file['x']
    z = mat_file['z']
    theta = mat_file['theta']
    vx = mat_file['vx']
    vz = mat_file['vz']
    q = mat_file['q']
    u = mat_file['u']
    w = mat_file['w']

    alpha = mat_file['alpha']
    gamma = mat_file['gamma']

    delta = mat_file['delta']

    t = mat_file['t'].squeeze()
    params = mat_file['params']

    n_trajectories = params.shape
    dt = t[1]-t[0]

    # x_train = np.vstack((
    #     x[0],
    #     z[0],
    #     theta[0],
    #     vx[0],
    #     vz[0],
    #     q[0]
    # )).T
    # feature_names = ['x', 'z', 'theta', 'vx', 'vz', 'q']

    # x_train = np.vstack((
    #     z[0],
    #     gamma[0],
    #     alpha[0],
    #     q[0],
    #     V[0]
    # )).T
    # feature_names = ['z', 'gamma', 'alpha', 'q', 'V']

    x_train = np.vstack((
        x[0],
        z[0],
        theta[0],
        u[0],
        w[0],
        q[0]
    )).T
    feature_names = ['x', 'z', 'theta', 'u', 'w', 'q']

    u_train = delta[0][:, np.newaxis]

    x_train = x_train[11:, :]
    u_train = u_train[11:, :]




    model = ps.SINDy(
        feature_library=custom_library,
        optimizer=ps.STLSQ(threshold=5e-3),
        feature_names=feature_names
    )
    # model.fit(x_train, u=u_train, t=dt)
    model.fit(x_train, t=dt)
    model.print()






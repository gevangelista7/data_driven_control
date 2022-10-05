
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from scipy.integrate import solve_ivp
from sklearn.linear_model import Lasso

import pysindy as ps

# bad code but allows us to ignore warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

np.random.seed(100)


def lorenz(t, X, sigma=10, beta=8/3, rho=28):
    """The Lorenz equations."""
    u, v, w = X
    up = -sigma*(u - v)
    vp = rho*u - v - u*w
    wp = -beta*w + u*v
    return up, vp, wp

integrator_keywords = {}
integrator_keywords['rtol'] = 1e-12
integrator_keywords['method'] = 'LSODA'
integrator_keywords['atol'] = 1e-12


if __name__ == '__main__':
    dt = .002

    t_train = np.arange(0, 10, dt)
    x0_train = [-8, 8, 27]
    t_train_span = (t_train[0], t_train[-1])
    x_train = solve_ivp(lorenz, t_train_span, x0_train,
                        t_eval=t_train, **integrator_keywords).y.T
    #%%
    # Instantiate and fit the SINDy model
    model = ps.SINDy()
    model.fit(x_train, t=dt)
    model.print()






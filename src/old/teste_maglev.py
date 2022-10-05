import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
from Maglev import Maglev


def generate_maglev(m=2.12e-2, g=9.8, y0=-7.47, gamma=328, i0=.514, rho=.166, K=1.2e-4):
    def maglev(t, x, u):
        x1, x2 = x
        dx1 = x2
        # u = 1 / rho * ((x1 - y0)/gamma * np.sqrt(m*g/K) - i0)
        # y = gamma * h + y0

        dx2 = gamma * g - K * (rho * u + i0) ** 2 * gamma ** 3 / m / ((x1 - y0) ** 2)
        return dx1, dx2

    return maglev


if __name__ == '__main__':
    # std_maglev = generate_maglev()
    #
    # maglev_hist = solve_ivp(
    #     std_maglev,
    #     t_span=(0, 1),
    #     y0=(-5, 0),
    #     args=(0, )
    # )
    #
    # plt.plot(maglev_hist.t, maglev_hist.y[0])
    # plt.plot(maglev_hist.t, maglev_hist.y[1])
    # plt.show()

    env = Maglev()
    s0 = env.reset()
    env.step(np.array(0.0))


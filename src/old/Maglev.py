from gym import logger, spaces
from gym import Env
import numpy as np
from typing import Optional


class Maglev(Env):
    def __init__(self):
        # Parameters from: Fama, Régis Campos, et al. "Predictive control of a magnetic levitation system with explicit treatment of operational constraints." Proceedings of the 18th International Congress of Mechanical Engineering, Ouro Preto, Brazil. 2005.
        self.m = 2.12e-2
        self.g = 9.8
        self.y0 = -7.47
        self.gamma = 328
        self.i0 = .514
        self.rho = .166
        self.K = 1.2e-4
        self.ref = .2

        self.y_threshold = 2
        self.dy_threshold = 50

        self.tau = 0.005
        self.max_steps = 1 / self.tau
        self.kinematics_integrator = 'euler'

        self.u_upper_threshold = 5
        self.u_lower_threshold = -3
        self.delta_u_limit = 50 * self.tau

        # y = gamma * h + y0
        # h_eq  = 0.022774 m
        # h_min = 0.016676 m
        # h_max = 0.288720 m

        high_act = np.array(
            [self.u_upper_threshold]
        )
        low_act = np.array(
            [self.u_lower_threshold]
        )
        high_obs = np.array(
            [
                self.y_threshold * 2,
                self.dy_threshold * 2
            ],
            dtype=np.float32
        )

        self.action_space = spaces.Box(low_act, high_act, dtype=np.float32)
        self.observation_space = spaces.Box(-high_obs, high_obs, dtype=np.float32)

        self.state = None
        self.steps_beyond_terminated = None
        self.previous_u = None
        self.step_counter = None
        self.control_discount = .9

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."

        #translation
        y_dot, y = self.state
        u = action

        # dynamic equation
        y_ddot = self.gamma * self.g - \
                 self.K * (self.rho * u + self.i0) ** 2 * self.gamma ** 3 /\
                 self.m / ((y - self.y0) ** 2)

        # euler integration
        y = y + self.tau * y_dot
        y_dot = y_dot + self.tau * y_ddot

        # save state
        self.state = [y_dot, y]

        # stop conditions
        terminated = bool(
            np.abs(y) > self.y_threshold
            or np.abs(y_dot) > self.dy_threshold
            or self.step_counter >= self.max_steps
        )

        reward = 1

        if not terminated:
            reward += -abs(y - self.ref)
        else:
            reward += -1

        self.step_counter += 1
        self.previous_u = u

        return np.array(self.state, dtype=np.float32), reward, terminated, {}

    def reset(self, *,
              seed: Optional[int] = None,
              custom_bound: Optional[float] = 1 / 2,
              control_discount=.5,
              ref=0):
        self.state = (np.random.uniform(low=-custom_bound * self.y_threshold,  high=custom_bound * self.y_threshold),
                      np.random.uniform(low=-custom_bound * self.dy_threshold, high=custom_bound * self.dy_threshold))
        self.steps_beyond_terminated = None
        self.control_discount = control_discount
        self.previous_u = 0
        self.step_counter = 0
        self.ref = ref

        return np.array(self.state, dtype=np.float32)

    def render(self, mode=None):
        pass

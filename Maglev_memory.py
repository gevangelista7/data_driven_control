from gym import Env, spaces
import numpy as np
from typing import Optional


class Maglev(Env):
    # implementation based on Cartpole version on GYM documentation
    # https://github.com/openai/gym/blob/master/gym/envs/classic_control/cartpole.py
    def __init__(self):
        # Parameters from: Fama, Régis Campos, et al.
        # "Predictive control of a magnetic levitation system with explicit treatment of operational constraints."
        # Proceedings of the 18th International Congress of Mechanical Engineering, Ouro Preto, Brazil. 2005.
        self.m = 2.12e-2
        self.g = 9.8
        self.y0 = -7.47
        self.gamma = 328
        self.i0 = .514
        self.rho = .166
        self.K = 1.2e-4

        self.y_threshold = 2
        self.dy_threshold = 50

        self.time_step = 0.005
        self.u_upper_threshold = 5
        self.u_lower_threshold = -3
        self.delta_u_limit = 3 / self.time_step

        self.max_steps = 1 / self.time_step

        # state with memory implementation like in Spielberg reference
        self.n_memory = 0

        # setup for gym environment framework
        high_act = np.array(
            [self.delta_u_limit]
        )
        high_obs = np.array(
            [
                self.y_threshold * 2,   # ref
                self.dy_threshold * 2,  # dy
                self.y_threshold * 2,   # y
                self.u_upper_threshold
            ] + self.n_memory * [
                self.y_threshold * 2,   # y_memory
            ],
            dtype=np.float32
        )

        low_obs = -high_obs
        low_obs[3] = self.u_lower_threshold

        self.action_space = spaces.Box(-high_act, high_act, dtype=np.float32)
        self.observation_space = spaces.Box(low_obs, high_obs, dtype=np.float32)

        # variables initialization
        self.state = None
        self.previous_u = None
        self.step_counter = None

        self.reference_function = None
        self.reward_function = None

    def step(self, action):
        err_msg = f"{action!r} ({type(action)}) invalid"
        assert self.action_space.contains(action), err_msg
        assert self.state is not None, "Call reset before using step method."
        assert self.reference_function is not None, "Define a reference function with config_reference method"
        assert self.reward_function is not None, "Define a reward function with config_reward method"

        # translate
        _, y_dot, y, u, *memory = self.state

        # u update
        u = self.previous_u + action * self.delta_u_limit
        u = np.clip(u, a_min=-3, a_max=5)

        # dynamic equation
        y_ddot = self.gamma * self.g - \
                 self.K * (self.rho * u + self.i0) ** 2 * self.gamma ** 3 / \
                 self.m / ((y - self.y0) ** 2)

        # memory shift
        if self.n_memory > 0:
            memory = [y] + memory[:-1]

        # euler integration
        y = y + self.time_step * y_dot
        y_dot = y_dot + self.time_step * y_ddot

        # get error
        ref = self.reference_function(self.step_counter * self.time_step)
        err = y - ref

        # save state internally
        self.state = err, y_dot, y, u, *memory

        # stopping condition
        terminated = bool(
            np.abs(y) > self.y_threshold
            or np.abs(y_dot) > self.dy_threshold
            or self.step_counter >= self.max_steps
        )

        # reward function
        if not terminated:
            reward = self.reward_function(self.state, action, ref)
        else:
            reward = 0

        self.step_counter += 1
        self.previous_u = u

        return np.array(self.state, dtype=np.float32).squeeze(), reward, terminated, {}

    def reset(self, *,
              seed: Optional[int] = None,
              x0_variance: Optional[float] = 0):
        self.state = (
            self.reference_function(0),
            np.random.uniform(low=-x0_variance * self.dy_threshold, high=x0_variance * self.dy_threshold),
            np.random.uniform(low=-x0_variance * self.y_threshold, high=x0_variance * self.y_threshold),
            0,
            *(self.n_memory * [0]))
        self.previous_u = 0
        self.step_counter = 0

        return np.array(self.state, dtype=np.float32)

    def render(self, mode=None):
        pass

    def config_reference(self, func):
        self.reference_function = func

    def config_reward(self, func):
        self.reward_function = func


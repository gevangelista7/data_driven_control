import gym
from gym import logger, spaces
from gym.error import DependencyNotInstalled
from gym import Env
from math import pi
from air import Atmosphere
import numpy as np
from numpy import interp
from typing import Optional, Union


class Aero3DOFEnv(Env):
    def __init__(self):

        self.mass = 500
        self.Jyy = 155
        self.Xcm = 1.3
        self.Xref = 1.3
        self.Lref = .36
        self.Sref = .1
        self.T = 15000

        self.dt = .01

        self.x = None
        self.z = None
        self.vx = None
        self.vz = None
        self.theta = None
        self.q = None

        self.x0 = 0
        self.z0 = -3000
        self.vx0 = 200
        self.vz0 = 0
        self.theta0 = 2*pi/180
        self.q0 = 0

        # TODO: Adequar ao CAMM e as interpolações
        # self.Cx_list =

        self.Cz0 = 0
        self.Cza = -5.6
        self.Czq = -42

        self.Cm0 = 0
        self.Cma = -6.8
        self.Cmq = -177

        self.air = Atmosphere()

        # TODO: adequar aos limites
        high = np.array(
            [
                self.x_threshold * 2,
                np.finfo(np.float32).max,
                self.theta_threshold_radians * 2,
                np.finfo(np.float32).max,
            ],
            dtype=np.float32,
        )

        self.action_space = spaces.Box(-9*pi/180, 9*pi/180, dtype=np.float32)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)


        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None
        self.isopen = True
        self.state = None
        # state = [ z, gamma, alpha, q, V, x]

    def Cx(self, mach):
        return interp(mach,
                      [0,       0.6,    .8,     .9,     1.0,    1.1,    1.2,    1.5,    2.0],
                      [-0.12,   -0.14,  -0.17,  -0.29  ,-0.49 , -0.55 , -0.52  ,-0.37 , -0.33 ])

    def Cm0(self, aoa, q, damping):
        pass

    def init_cond(self, x0, z0, vx0, vz0, theta0, q0):
        self.x0 = x0
        self.z0 = z0
        self.vx0 = vx0
        self.vz0 = vz0
        self.theta0 = theta0
        self.q0 = q0

    def reset(self):
        self.x = self.x0
        self.z = self.z0
        self.vx = self.vx0
        self.vz = self.vz0
        self.theta = self.theta0
        self.q = self.q0

    def step(self, action):
        z, gamma, alpha, q, V, x = self.state
        temp, pressure, density, sound_speed = self.air.get_conditions(-z)

        dynpress = .5*V*density*V**2
        mach = V / sound_speed
        damping = self.Lref / (2 * V)









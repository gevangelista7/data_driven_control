import numpy as np
from math import exp, sqrt


class Atmosphere:
    def __init__(self, deltaISA=0):
        self.deltaISA = deltaISA
        self.gamma = 1.4
        self.R = 8.314472 / 0.029
        self.p0 = 101325
        self.Tmsl = 288.15
        self.T0 = self.Tmsl + self.deltaISA

        self.g = 9.80665

    def get_conditions(self, alt):
        if alt <= 11000:
            temp = self.T0 - 0.0065*alt
            pressure = self.p0 * (temp /self.T0) ** (self.g/0.0064/self.R)

        elif alt <= 20000:
            temp = self.T0 - 71.55
            p11000 = self.p0 * ((temp/self.T0)**(self.g / (0.0065*self.R)))
            pressure = p11000 * exp(-self.g * (alt-11000) / (self.R*temp))

        else:
            assert False, 'Modelo atmosférico não definido acima de 20 km'

        density = pressure / self.R * temp
        sound_speed = sqrt( self.gamma*self.R*temp)

        return temp, pressure, density, sound_speed









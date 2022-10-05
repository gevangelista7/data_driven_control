import numpy as np
from scipy.integrate import solve_ivp
from sklearn.svm import SVC

class Detector:
    def __init__(self, n_state, n_frontier, stride=1):
        assert n_frontier > n_state, "n_frontier must be greater then n_state"
        self.n_state = n_state
        self.n_frontier = n_frontier
        self.stride = stride
        self.frontier_samples = None

    def bootstrap(self, frontier_sample):
        samples = []
        for i in range(self.n_state, self.n_frontier, self.stride):
            samples.append(frontier_sample[i-self.n_state: i])

        self.frontier_samples = np.array(samples)

    # def




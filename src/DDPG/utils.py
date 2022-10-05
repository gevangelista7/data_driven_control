import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curve(x, scores, figure_file):
    running_avg = np.zeros(len(scores))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(scores[max(0, i-100):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.savefig(figure_file)


class OrnsteinUhlenbeckNoise:
    # https://math.stackexchange.com/questions/1287634/implementing-ornstein-uhlenbeck-in-matlab
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.mu = mu
        self.sigma = sigma
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        self.x_t = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __call__(self):
        x_next = self.x_t + \
                 self.theta * (self.mu - self.x_t) * self.dt + \
                 self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)

        self.x_t = x_next
        return x_next


class ReplayBuffer:
    def __init__(self, max_size, state_shape, n_actions):
        self.max_size = max_size

        self.state_memory = np.zeros((self.max_size, *state_shape))
        self.action_memory = np.zeros((self.max_size, n_actions))
        self.next_state_memory = np.zeros((self.max_size, *state_shape))
        self.reward_memory = np.zeros(self.max_size)
        self.done_memory = np.zeros(self.max_size, dtype=np.bool)

        self.ptr = 0

    def update_pointer(self):
        self.ptr += 1

    def store_transition(self, state, action, next_state, reward, done):
        idx = self.ptr % self.max_size

        self.state_memory[idx] = state
        self.action_memory[idx] = action
        self.next_state_memory[idx] = next_state
        self.reward_memory[idx] = reward
        self.done_memory[idx] = done

        self.update_pointer()

    def sample_memory(self, batch_size):
        max_idx = min(self.ptr, self.max_size)
        if batch_size > max_idx:
            raise "Sample size is too big"

        indexes = np.random.choice(max_idx, batch_size, replace=False)

        state_sample = self.state_memory[indexes]
        action_sample = self.action_memory[indexes]
        next_state_sample = self.next_state_memory[indexes]
        reward_sample = self.reward_memory[indexes]
        done_sample = self.done_memory[indexes]

        return state_sample, action_sample, next_state_sample, reward_sample, done_sample
        # states, actions, next_states, rewards, dones
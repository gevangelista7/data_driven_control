import os
import numpy as np
import torch as t
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as f


class CriticNetwork(nn.Module):
    def __init__(self, obs_shape, n_action,
                 lr=1e-3, fc1dim=400, fc2dim=300,
                 weight_decay=1e-1, device='cpu',
                 filename='test', checkpoint_dir='models/'):
        super(CriticNetwork, self).__init__()

        self.obs_shape = obs_shape
        self.n_action = n_action

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, filename)

        self.bn1 = nn.LayerNorm(fc1dim)
        self.bn2 = nn.LayerNorm(fc2dim)

        self.fc1 = nn.Linear(*obs_shape, fc1dim)
        f_in = float(1/np.sqrt(np.prod(obs_shape)))
        self.fc1.weight.data.uniform_(-f_in, f_in)
        self.fc1.bias.data.uniform_(f_in, f_in)
        self.bn1 = nn.LayerNorm(fc1dim)

        self.fc2 = nn.Linear(fc1dim, fc2dim)
        f_in2 = float(1/np.sqrt(fc1dim))
        self.fc2.weight.data.uniform_(-f_in2, f_in2)
        self.fc2.bias.data.uniform_(-f_in2, f_in2)
        self.bn2 = nn.LayerNorm(fc2dim)

        self.q_layer = nn.Linear(fc2dim, 1)
        self.q_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.q_layer.bias.data.uniform_(-3e-3, 3e-3)

        self.a_layer = nn.Linear(n_action, fc2dim)
        self.a_layer.weight.data.uniform_(-3e-3, 3e-3)
        self.a_layer.bias.data.uniform_(-3e-3, 3e-3)

        self.optimizer = optim.Adam(params=self.parameters(), lr=lr, weight_decay=weight_decay)
        self.device = device
        self.to(device)

    def forward(self, state, action):
        state_value = self.fc1(state)
        state_value = f.relu(self.bn1(state_value))

        state_value = self.fc2(state_value)
        state_value = f.relu(self.bn2(state_value))

        action_value = self.a_layer(action)

        state_action_value = f.relu(action_value + state_value)
        q = self.q_layer(state_action_value)

        return q

    def save_checkpoint(self):
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        self.load_state_dict(t.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        t.save(self.state_dict(), checkpoint_file)


class ActorNetwork(nn.Module):
    def __init__(self, obs_shape, n_action,
                 lr=1e-3, fc1dim=400, fc2dim=300,
                 device='cpu',
                 filename='test', checkpoint_dir='models/'):
        super(ActorNetwork, self).__init__()

        self.obs_shape = obs_shape
        self.n_action = n_action

        self.checkpoint_dir = checkpoint_dir
        self.checkpoint_file = os.path.join(self.checkpoint_dir, filename)

        self.fc1 = nn.Linear(*obs_shape, fc1dim)
        f_in = float(1/np.sqrt(np.prod(obs_shape)))
        self.fc1.weight.data.uniform_(-f_in, f_in)
        self.fc1.bias.data.uniform_(f_in, f_in)
        self.bn1 = nn.LayerNorm(fc1dim)

        self.fc2 = nn.Linear(fc1dim, fc2dim)
        f_in2 = float(1/np.sqrt(fc1dim))
        self.fc2.weight.data.uniform_(-f_in2, f_in2)
        self.fc2.bias.data.uniform_(-f_in2, f_in2)
        self.bn2 = nn.LayerNorm(fc2dim)

        self.mu = nn.Linear(fc2dim, n_action)
        self.mu.weight.data.uniform_(-3e-3, 3e-3)
        self.mu.bias.data.uniform_(-3e-3, 3e-3)

        self.optimizer = optim.Adam(params=self.parameters(), lr=lr)
        self.device = device
        self.to(device)

    def forward(self, state):
        x = self.fc1(state)
        x = f.relu(self.bn1(x))

        x = self.fc2(x)
        x = f.relu(self.bn2(x))

        x = self.mu(x)
        actions = t.tanh(x)
        # actions = 4 * t.tanh(x) + 1
        return actions

    def save_checkpoint(self):
        print('... saving checkpoint ...')
        t.save(self.state_dict(), self.checkpoint_file)

    def load_checkpoint(self):
        print('... loading checkpoint ...')
        self.load_state_dict(t.load(self.checkpoint_file))

    def save_best(self):
        print('... saving best checkpoint ...')
        checkpoint_file = os.path.join(self.checkpoint_dir, self.name+'_best')
        t.save(self.state_dict(), checkpoint_file)





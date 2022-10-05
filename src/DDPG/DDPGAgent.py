
import torch as t
import numpy as np
import torch.nn.functional as f
from src.DDPG.DDPGNetworks import ActorNetwork, CriticNetwork
from src.DDPG.utils import OrnsteinUhlenbeckNoise, ReplayBuffer


class DDPGAgent:
    def __init__(self, obs_shape, n_action,
                 lr_actor=1e-4, lr_critic=1e-3,
                 tau=1e-3, gamma=.99,
                 replay_buffer_size=1e6, batch_size=64,
                 uo_theta=.15, uo_sigma=.2,
                 device='cpu', fc1dim=400, fc2dim=300,
                 critic_l2=1e-2):

        self.obs_shape = obs_shape
        self.n_action = n_action
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.tau = tau
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size

        self.std_uo_sigma = uo_sigma
        self.std_uo_theta = uo_theta

        self.memory = ReplayBuffer(max_size=int(replay_buffer_size),
                                   state_shape=[obs_shape],
                                   n_actions=n_action)

        self.noise = OrnsteinUhlenbeckNoise(mu=np.zeros(n_action),
                                            sigma=uo_sigma, theta=uo_theta)

        self.actor_net = ActorNetwork(obs_shape=[obs_shape], n_action=n_action, lr=lr_actor,
                                      fc1dim=fc1dim, fc2dim=fc2dim, device=device, filename='actor_ddpg')

        self.critic_net = CriticNetwork(obs_shape=[obs_shape], n_action=n_action, lr=lr_critic,
                                        fc1dim=fc1dim, fc2dim=fc2dim, device=device, weight_decay=critic_l2, filename='critic_ddpg')

        self.actor_net_target = ActorNetwork(obs_shape=[obs_shape], n_action=n_action, lr=lr_actor,
                                             fc1dim=fc1dim, fc2dim=fc2dim, device=device, filename='actor_tgt_ddpg')

        self.critic_net_target = CriticNetwork(obs_shape=[obs_shape], n_action=n_action, lr=lr_critic,
                                               fc1dim=fc1dim, fc2dim=fc2dim, device=device, weight_decay=critic_l2, filename='critic_tgt_ddpg')

        self.update_network_parameters(tau=1)

    def choose_action(self, observation):
        self.actor_net.eval()
        state = t.tensor(observation, dtype=t.float, device=self.actor_net.device)
        action = self.actor_net.forward(state) + t.tensor(self.noise(), dtype=t.float, device=self.actor_net.device)
        self.actor_net.train()
        return action.cpu().detach().numpy()

    def store_transition(self, state, action, next_state, reward, done):
        self.memory.store_transition(state, action, next_state, reward, done)

    def sample_memory(self):
        states, actions, states_, rewards, dones = self.memory.sample_memory(self.batch_size)
        # states, actions, next_states, rewards, dones

        states = t.tensor(states, device=self.device, dtype=t.float)
        actions = t.tensor(actions, device=self.device, dtype=t.float)
        rewards = t.tensor(rewards, device=self.device, dtype=t.float)
        states_ = t.tensor(states_, device=self.device, dtype=t.float)
        dones = t.tensor(dones, device=self.device, dtype=t.bool)

        return states, actions, states_, rewards, dones
        # states, actions, next_states, rewards, dones

    def save_checkpoints(self):
        self.critic_net.save_checkpoint()
        self.critic_net_target.save_checkpoint()
        self.actor_net.save_checkpoint()
        self.actor_net_target.save_checkpoint()

    def load_checkpoint(self):
        self.critic_net.load_checkpoint()
        self.critic_net_target.load_checkpoint()
        self.actor_net.load_checkpoint()
        self.actor_net_target.load_checkpoint()

    def learn(self):
        # sampling
        if self.memory.ptr < self.batch_size:
            print('skipping learning, not enough transitions...')
            return

        states, actions, next_states, rewards, dones = self.sample_memory()

        tgt_actions = self.actor_net_target.forward(state=next_states)
        q_next_tgt = self.critic_net_target.forward(state=next_states, action=tgt_actions)
        q_next_tgt[dones] = 0.0

        q_tgt = rewards + self.gamma * q_next_tgt.view(-1)
        q_tgt = q_tgt.view(self.batch_size, 1)

        q_pred = self.critic_net.forward(states, actions)

        self.critic_net.optimizer.zero_grad()
        critic_loss = f.mse_loss(input=q_tgt, target=q_pred).to(self.critic_net.device)
        critic_loss.backward()
        self.critic_net.optimizer.step()

        self.actor_net.optimizer.zero_grad()
        actor_loss = t.mean(-self.critic_net.forward(states, self.actor_net.forward(states)))
        actor_loss.backward()
        self.actor_net.optimizer.step()

        self.update_network_parameters(self.tau)

    def update_network_parameters(self, tau):
        actor_params = self.actor_net.state_dict()
        critic_params = self.critic_net.state_dict()
        tgt_actor_params = self.actor_net_target.state_dict()
        tgt_critic_params = self.critic_net_target.state_dict()

        for key in tgt_actor_params.keys():
            tgt_actor_params[key] = tau * actor_params[key].clone() + (1 - tau) * tgt_actor_params[key].clone()

        for key in tgt_critic_params.keys():
            tgt_critic_params[key] = tau * critic_params[key].clone() + (1 - tau) * tgt_critic_params[key].clone()

        self.actor_net_target.load_state_dict(tgt_actor_params)
        self.critic_net_target.load_state_dict(tgt_critic_params)

    # adaptation to Spielberg work
    def learning_update(self, state, tol):
        set_point = state[0]        # the first state position is dedicated for reference_t
        memory = state[2:]          # state[2:] -> y_t to t_{t-memory_len}
        last_states_into_tol = (np.abs(memory - set_point) < tol).all()
        if last_states_into_tol:
            self.noise.sigma = 0
            self.noise.theta = 0
        else:
            self.noise.sigma = self.std_uo_sigma
            self.noise.theta = self.std_uo_theta

    def define_networks_filenames(self, filename):
        self.actor_net.checkpoint_file = filename + 'actor'
        self.critic_net.checkpoint_file = filename + 'critic'
        self.actor_net_target.checkpoint_file = filename + 'actor_tgt'
        self.critic_net_target.checkpoint_file = filename + 'critic_tgt'





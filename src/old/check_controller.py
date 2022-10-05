import torch
import numpy as np
from Maglev_memory import Maglev
from src.DDPG.DDPGAgent import DDPGAgent
import matplotlib.pyplot as plt
plt.rcParams['axes.grid'] = True

SCORE_UPDATE = 1
N_GAMES = 1

TRAINING = False


def const_ref(t):
    return .2


def exponential_deviation_rwd(state, action, ref):
    y = state[2]
    return np.exp(-np.abs(y - ref)) # / np.abs(ref))


env = Maglev()
env.config_reference(const_ref)
env.config_reward(exponential_deviation_rwd)

if __name__ == "__main__":
    state_shape = env.observation_space.shape
    device = "cuda" if torch.cuda.is_available() else "cpu"

    agent = DDPGAgent(obs_shape=env.observation_space.shape[0],
                      n_action=env.action_space.shape[0],
                      lr_actor=1e-4, lr_critic=1e-3,
                      tau=1e-3, gamma=.99,
                      replay_buffer_size=1e6, batch_size=64,
                      uo_theta=.0, uo_sigma=.0,
                      device=device, fc1dim=400, fc2dim=300,
                      critic_l2=1e-1)

    agent.actor_net.checkpoint_file = '../../models/odl/actor_ddpg'
    agent.actor_net_target.checkpoint_file = '../../models/odl/actor_tgt_ddpg'
    agent.critic_net.checkpoint_file = '../../models/odl/critic_ddpg'
    agent.critic_net_target.checkpoint_file = '../../models/odl/critic_tgt_ddpg'
    agent.load_checkpoint()

    states = []
    actions = []
    rewards = []

    print("Device: ", agent.device, "\n")
    for i in range(N_GAMES):
        obs = env.reset()
        game_score = 0
        done = False

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.learning_update(next_obs, 0.01)
            # agent.store_transition(state=obs, action=action, next_state=next_obs, reward=reward, done=done)
            # agent.learn()

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs

    t = env.time_step * np.arange(min(len(states), env.max_steps+1))
    states = np.array(states)

    fig, ax = plt.subplots(2)

    ax[0].plot(t, states.T[0, :], 'r--', label='ref')
    ax[0].plot(t, states.T[2, :], label='y')
    ax[0].legend()
    # ax[1].plot(t, states.T[1, :], label='dy/dt')
    # ax[1].legend()
    ax[1].plot(t, actions, label='u')
    ax[1].legend()
    # ax[2].plot(t, np.cumsum(rewards), label='rwd')
    # ax[2].legend()

    plt.show()
    print('total rwd: {}'.format(np.cumsum(rewards)[-1].item()))

    run_info = {
        't': t,
        'states': states,
        'actions': actions,
        'rewards': rewards
    }
    # joblib.dump(run_info, "results/step_result_err_fac_{}_u_fac{}'.format(env.err_factor, env.control_factor)")

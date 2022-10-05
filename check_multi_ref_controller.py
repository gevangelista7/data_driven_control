import torch
import numpy as np
from Maglev_memory import Maglev
from src.DDPG.DDPGAgent import DDPGAgent
import matplotlib.pyplot as plt
from reward_func_lib import *
plt.rcParams['axes.grid'] = True

SCORE_UPDATE = 1
N_GAMES = 1

TRAINING = False


def generate_const_ref_func(r):
    def const_ref(t):
        return r

    return const_ref

env = Maglev()
check_values = [.9, .5, .25, -.25, -.5, .9]
quad_err = []

fig, ax = plt.subplots(len(check_values), 1)
suffix = '_opt2'

if __name__ == "__main__":
    env.config_reward(quadratic_err)
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

    agent.actor_net.checkpoint_file = 'models/actor_ddpg' + suffix
    agent.actor_net_target.checkpoint_file = 'models/actor_tgt_ddpg' + suffix
    agent.critic_net.checkpoint_file = 'models/critic_ddpg' + suffix
    agent.critic_net_target.checkpoint_file = 'models/critic_tgt_ddpg' + suffix
    agent.load_checkpoint()

    for i in range(len(check_values)):
        env.config_reference(generate_const_ref_func(check_values[i]))
        obs = env.reset()
        game_score = 0
        done = False
        states = []
        actions = []
        rewards = []

        while not done:
            action = agent.choose_action(obs)
            next_obs, reward, done, _ = env.step(action)
            agent.learning_update(next_obs, 0.5)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs

        t = env.time_step * np.arange(min(len(states), env.max_steps+1))
        states = np.array(states)

        ax[i].plot(t, states.T[2, :], label='y')
        ax[i].plot(t, [env.reference_function(t_i) for t_i in t], label='ref')
        # ax[i, 1].plot(t, states.T[0, :], 'r--', label='err')

        quad_err.append(np.mean(rewards).item())

    ax[0].legend()
    # ax[0, 1].legend()
    fig.show()
    print('Quadratic mean err: {} ({})'.format(quad_err, np.mean(quad_err)))

    # run_info = {
    #     't': t,
    #     'states': states,
    #     'actions': actions,
    #     'rewards': rewards
    # }
    # joblib.dump(run_info, "results/step_result_err_fac_{}_u_fac{}'.format(env.err_factor, env.control_factor)")


import joblib
import numpy as np
from matplotlib import pyplot as plt

from src.DDPG.DDPGAgent import DDPGAgent
from src.DDPG.utils import plot_learning_curve
from PaperMachine import PaperMachine
from reward_func_lib import *
plt.rcParams['axes.grid'] = True
n_episodes = 400

env = PaperMachine()
env.config_reward(neg_abs_err_rwd)
env.config_reference(generate_const_ref_func(0))


if __name__ == '__main__':

    agent = DDPGAgent(obs_shape=env.observation_space.shape[0],
                      n_action=env.action_space.shape[0],
                      lr_actor=1e-4, lr_critic=1e-3,
                      tau=1e-3, gamma=.99,
                      replay_buffer_size=5e4, batch_size=256,
                      uo_theta=.15, uo_sigma=.1,
                      device='cuda', fc1dim=400, fc2dim=300,
                      critic_l2=1e-4)

    agent.actor_net.checkpoint_file = 'models/odl/actor_ddpg_pm'
    agent.actor_net_target.checkpoint_file = 'models/odl/actor_tgt_ddpg_pm'
    agent.critic_net.checkpoint_file = 'models/odl/critic_ddpg_pm'
    agent.critic_net_target.checkpoint_file = 'models/odl/critic_tgt_ddpg_pm'
    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_episodes):
        obs = env.reset(x0_variance=0.0)
        done = False
        score = 0
        agent.noise.reset()
        r = np.random.uniform(0, 10)
        # r = .2
        env.config_reference(generate_const_ref_func(r))
        while not done:
            action = agent.choose_action(obs)
            action = np.array([max(min(env.u_upper_threshold, action.item()), env.u_lower_threshold)], dtype=np.float64)
            next_obs, reward, done, _ = env.step(action)
            agent.store_transition(obs, action, next_obs, reward, done)
            agent.learning_update(next_obs, 0.01)
            agent.learn()
            score += reward
            obs = next_obs

        score_history.append(score)
        avg_score = np.mean(score_history[-21:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()
        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)

    x = [i + 1 for i in range(n_episodes)]
    running_avg = np.zeros(len(score_history))
    for i in range(len(running_avg)):
        running_avg[i] = np.mean(score_history[max(0, i-21):(i+1)])
    plt.plot(x, running_avg)
    plt.title('Running average of previous 100 scores')
    plt.show()

    agent.load_checkpoint()
    check_values = [2, 4, 11, 15]
    fig, ax = plt.subplots(len(check_values))

    for i in range(len(check_values)-1):
        ax[i].tick_params(bottom=False)
        ax[i].tick_params(bottom=False)

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
            agent.learning_update(next_obs, 100)

            states.append(obs)
            actions.append(action)
            rewards.append(reward)
            obs = next_obs

        t = env.time_step * np.arange(min(len(states), env.max_steps+1))
        states = np.array(states)

        ax[i].plot(t, states.T[1, :], label='y')
        ax[i].plot(t, [env.reference_function(t_i) for t_i in t], 'r--', label='ref')
        print('total rwd: {}'.format(np.cumsum(rewards)[-1].item()))

    ax[0].legend()
    ax[0].legend()
    fig.show()

    fig, ax = plt.subplots(1)

    env.config_reference(multi_step_ref_func)
    obs = env.reset()
    game_score = 0
    done = False
    states = []
    actions = []
    rewards = []

    while not done:
        action = agent.choose_action(obs)
        next_obs, reward, done, _ = env.step(action)
        agent.learning_update(next_obs, 100)

        states.append(obs)
        actions.append(action)
        rewards.append(reward)
        obs = next_obs

    t = env.time_step * np.arange(min(len(states), env.max_steps + 1))
    states = np.array(states)

    ax.plot(t, states.T[1, :], label='y')
    ax.plot(t, [env.reference_function(t_i) for t_i in t], 'r--', label='ref')
    print('total rwd: {}'.format(np.cumsum(rewards)[-1].item()))
    ax.legend()
    fig.show()

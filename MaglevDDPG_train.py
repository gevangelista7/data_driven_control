import joblib
import numpy as np
from src.DDPG.DDPGAgent import DDPGAgent
from src.DDPG.utils import plot_learning_curve
from Maglev_memory import Maglev
from reward_func_lib import *

n_episodes = 50000

env = Maglev()
env.config_reward(optim_exp_dev_rwd)
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

    agent.actor_net.checkpoint_file = 'models/secundarios/actor_ddpg_opt2_mem'
    agent.actor_net_target.checkpoint_file = 'models/secundarios/actor_tgt_ddpg_opt2_mem'
    agent.critic_net.checkpoint_file = 'models/secundarios/critic_ddpg_opt2_mem'
    agent.critic_net_target.checkpoint_file = 'models/secundarios/critic_tgt_ddpg_opt2_mem'
    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_episodes):
        obs = env.reset(x0_variance=0.0)
        done = False
        score = 0
        agent.noise.reset()
        r = np.random.uniform(-.5, .5)
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
        avg_score = np.mean(score_history[-100:])

        if avg_score > best_score:
            best_score = avg_score
            agent.save_checkpoints()
        print('episode ', i, 'score %.1f' % score,
              'average score %.1f' % avg_score)

    x = [i + 1 for i in range(n_episodes)]

    filename = 'LunarLander_lr_act_' + str(agent.lr_actor) + '_lr_crt_' + \
               str(agent.lr_critic) + '_' + str(n_episodes) + '_games'
    filename += 'deterministico_contr_disc=.8'
    figure_file = 'plots/' + filename + '.png'
    plot_learning_curve(x, score_history, figure_file)

    joblib.dump(score_history, 'results/learning_rwd_opt')



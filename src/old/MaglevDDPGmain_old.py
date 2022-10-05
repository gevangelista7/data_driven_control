import numpy as np
from src.DDPG.DDPGAgent import DDPGAgent
from src.DDPG.utils import plot_learning_curve
from Maglev import Maglev

env = Maglev()

if __name__ == '__main__':

    agent = DDPGAgent(obs_shape=env.observation_space.shape[0],
                      n_action=env.action_space.shape[0],
                      lr_actor=1e-4, lr_critic=1e-3,
                      tau=1e-3, gamma=.99,
                      replay_buffer_size=5e4, batch_size=128,
                      uo_theta=.15, uo_sigma=.03,
                      device='cuda', fc1dim=400, fc2dim=300,
                      critic_l2=1e-4)

    agent.actor_net.checkpoint_file = 'models/actor_ddpg_err_abs'
    agent.actor_net_target.checkpoint_file = 'models/actor_tgt_ddpg_optim_err_abs'
    agent.critic_net.checkpoint_file = 'models/critic_ddpg_optim_err_abs'
    agent.critic_net_target.checkpoint_file = 'models/critic_tgt_ddpg_optim_err_abs'

    n_games = 3000
    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        obs = env.reset(ref=0.2, custom_bound=0.0, control_discount=.8)
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(obs)
            action = np.array([max(min(env.u_upper_threshold, action.item()), env.u_lower_threshold)], dtype=np.float32)
            next_obs, reward, done, _ = env.step(action)
            agent.store_transition(obs, action, next_obs, reward, done)
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

    x = [i + 1 for i in range(n_games)]

    filename = 'LunarLander_lr_act_' + str(agent.lr_actor) + '_lr_crt_' + \
                str(agent.lr_critic) + '_' + str(n_games) + '_games'
    filename += 'deterministico_contr_disc=.8'
    figure_file = 'plots/' + filename + '.png'
    plot_learning_curve(x, score_history, figure_file)



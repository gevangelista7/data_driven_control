import joblib
import numpy as np
from src.DDPG.DDPGAgent import DDPGAgent
from src.DDPG.utils import plot_learning_curve
from src.old.Maglev_memory_old import Maglev

env = Maglev()
n_games = 500
err_factor = 1
control_factor = 0
rwd_baseline = 0

if __name__ == '__main__':

    agent = DDPGAgent(obs_shape=env.observation_space.shape[0],
                      n_action=env.action_space.shape[0],
                      lr_actor=1e-4, lr_critic=1e-3,
                      tau=1e-3, gamma=.99,
                      replay_buffer_size=5e4, batch_size=128,
                      uo_theta=.15, uo_sigma=.01,
                      device='cuda', fc1dim=400, fc2dim=300,
                      critic_l2=1e-4)

    best_score = env.reward_range[0]
    score_history = []

    for i in range(n_games):
        ref_var = np.random.uniform(0, .5)
        obs = env.reset(ref=ref_var,
                        x0_variance=0.0,
                        err_factor=err_factor,
                        control_factor=control_factor,
                        rwd_baseline=rwd_baseline)
        done = False
        score = 0
        agent.noise.reset()
        while not done:
            action = agent.choose_action(obs)
            action = np.array([max(min(env.u_upper_threshold, action.item()), env.u_lower_threshold)], dtype=np.float32)
            next_obs, reward, done, _ = env.step(action)
            agent.store_transition(obs, action, next_obs, reward, done)
            agent.learning_update(next_obs, env.ref, 0.01)
            agent.learn()
            score += reward
            obs = next_obs

        score_history.append(score)
        avg_score = np.mean(score_history[-10:])

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

    joblib.dump(score_history, 'results/learning_err_fac_{}_u_fac{}'.format(env.err_factor, env.control_factor))



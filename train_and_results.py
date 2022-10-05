from Maglev_memory import Maglev
from reward_func_lib import *
from src.DDPG.DDPGAgent import DDPGAgent
from Coach import Coach

device = 'cuda'


if __name__ == '__main__':
    n_episodes = 100
    env_rwd_1 = Maglev()
    env_rwd_1.config_reward(exponential_deviation_rwd)
    env_rwd_1.config_reference(generate_const_ref_func(0))

    agent_1 = DDPGAgent(obs_shape=env_rwd_1.observation_space.shape[0],
                      n_action=env_rwd_1.action_space.shape[0],
                      lr_actor=1e-4, lr_critic=1e-3,
                      tau=1e-3, gamma=.99,
                      replay_buffer_size=5e4, batch_size=256,
                      uo_theta=.15, uo_sigma=.1,
                      device=device, fc1dim=400, fc2dim=300,
                      critic_l2=1e-4)

    coach_1 = Coach(env=env_rwd_1, agent=agent_1, n_episodes=n_episodes, filename='exponential_deviation_rwd')
    coach_1.train()



    env_rwd_2 = Maglev()
    env_rwd_2.config_reward(exp_discontinuous_rwd)
    env_rwd_2.config_reference(generate_const_ref_func(0))

    agent_2 = DDPGAgent(obs_shape=env_rwd_2.observation_space.shape[0],
                      n_action=env_rwd_2.action_space.shape[0],
                      lr_actor=1e-4, lr_critic=1e-3,
                      tau=1e-3, gamma=.99,
                      replay_buffer_size=5e4, batch_size=256,
                      uo_theta=.15, uo_sigma=.1,
                      device=device, fc1dim=400, fc2dim=300,
                      critic_l2=1e-4)

    coach_2 = Coach(env=env_rwd_2, agent=agent_2, n_episodes=n_episodes, filename='exponential_deviation_rwd')
    coach_2.train()


    env_rwd_3 = Maglev()
    env_rwd_3.config_reward(optim_exp_dev_rwd)
    env_rwd_3.config_reference(generate_const_ref_func(0))

    agent_3 = DDPGAgent(obs_shape=env_rwd_3.observation_space.shape[0],
                      n_action=env_rwd_3.action_space.shape[0],
                      lr_actor=1e-4, lr_critic=1e-3,
                      tau=1e-3, gamma=.99,
                      replay_buffer_size=5e4, batch_size=256,
                      uo_theta=.15, uo_sigma=.1,
                      device=device, fc1dim=400, fc2dim=300,
                      critic_l2=1e-4)

    coach_3 = Coach(env=env_rwd_3, agent=agent_3, n_episodes=n_episodes, filename='exponential_deviation_rwd')
    coach_3.train()
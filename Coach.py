import numpy as np
import joblib
from reward_func_lib import *

class Coach:
    def __init__(self, env, agent, n_episodes, filename):
        self.env = env
        self.agent = agent
        self.n_episodes = n_episodes
        self.filename = filename

        agent.actor_net.checkpoint_file = 'models/' + filename + 'actor'
        agent.actor_net_target.checkpoint_file = 'models/' + filename + 'actor_tgt'
        agent.critic_net.checkpoint_file = 'models/' + filename + 'critic'
        agent.critic_net_target.checkpoint_file = 'models/' + filename + 'critic_tgt'

    def train(self):
        best_score = self.env.reward_range[0]
        score_history = []

        # run n_episodes
        for i in range(self.n_episodes):
            # reset the enviroment, done variable, score, and noise
            obs = self.env.reset(x0_variance=0.0)
            done = False
            score = 0
            self.agent.noise.reset()

            # define the level of step at the beginning of each episode and set the const ref function with generator function
            # the training will be done with steps in uniformly sampled levels
            r = np.random.uniform(-.5, .5)
            self.env.config_reference(generate_const_ref_func(r))

            # run the episode. done variable finnish it, the reward of each step is summed in score
            while not done:
                action = self.agent.choose_action(obs)
                next_obs, reward, done, _ = self.env.step(action)
                self.agent.store_transition(obs, action, next_obs, reward, done)
                self.agent.learning_update(next_obs, 0.01)
                self.agent.learn()
                score += reward
                obs = next_obs

            # at the end of each episode the total score is registered
            score_history.append(score)
            # the moving average of 100 episodes is used to compare the models
            avg_score = np.mean(score_history[-100:])

            # if the moving avg is the best save the model into a file
            if avg_score > best_score:
                best_score = avg_score
                self.agent.save_checkpoints()
            print('episode ', i, 'score %.1f' % score,
                  'average score %.1f' % avg_score)

        # get the best model from file
        self.agent.load_checkpoint()

        # store the learning function on a file
        result = {
            'score_history': score_history,
            'best_score': best_score
        }
        joblib.dump(result, 'result/'+self.filename+'_learning')

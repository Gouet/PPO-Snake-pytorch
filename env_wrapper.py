import torch
import numpy as np
import pySnake

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

class EnvWrapper:
    def __init__(self, gym_env, actors, saved_episode, update_obs=None, update_reward=None, end_episode=None):
        self.envs = []
        self.variables = []
        self.update_obs = update_obs
        self.episode = 0
        self.end_episode = end_episode
        self.update_reward = update_reward
        self.saved_episode = saved_episode
        self.global_step = 0
        self.episode_step = []
        self.wall_size = 6
        self.factor_size = 0
        self.can_saved = False
        self.scenario = gym_env
        for _ in range(actors):
            env = pySnake.make()
            self.action_shape = 3
            self.upper_bound = 0
            self.continious = False
            self.envs.append(env)
        for _ in range(actors):
            self.variables.append([])
            self.episode_step.append(0)

    def add_variables_at_index(self, id, data):
        self.variables[id] = data

    def get_variables_at_index(self, id):
        return self.variables[id]

    def step(self, actions):
        batch_states = []
        batch_rewards = []
        batch_dones = []
        self.can_saved = False

        for i, action in enumerate(actions):
            self.episode_step[i] += 1
            states, rewards, done_ = self.envs[i].step(action) # action
            if done_ == True:
                if self.episode % 5500 == 0 and self.episode > 0:
                    self.reduce_wall_size()
                states = self.envs[i].reset()
                if self.episode % self.saved_episode == 0:
                    self.can_saved = True
                if self.end_episode is not None and done_:
                    self.episode += 1
                    self.end_episode(self, self.episode, self.variables[i], self.global_step, self.episode_step[i])
                    self.episode_step[i] = 0
                    self.variables[i] = []
            if self.update_reward is not None:
                rewards = self.update_reward(rewards)
            if self.update_obs is not None:
                states = self.update_obs(states)
            batch_states.append(states)
            batch_rewards.append(rewards)
            batch_dones.append(done_)
        self.dones = batch_dones
        self.global_step += 1
        return batch_states, batch_rewards, batch_dones

    def render(self):
        pySnake.render(True)

    def done(self):
        return all(self.dones)

    def set_wall_size(self, size):
        pySnake.reduce_wall(size)

    def reduce_wall_size(self):
        self.wall_size -= 1
        self.factor_size += 1
        if self.wall_size <= 0:
            self.wall_size = 1
        #pySnake.reduce_wall(self.wall_size)

    def reset(self):
        batch_states = []
        self.dones = []
        print('RESET')
        for env in self.envs:
            obs = env.reset()
            self.dones.append(False)
            if self.update_obs is not None:
                obs = self.update_obs(obs)
            batch_states.append(obs)
        return batch_states
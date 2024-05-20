from envs.multiagentenv import MultiAgentEnv
from utils.dict2namedtuple import convert
import numpy as np
import torch as th

from .multiagent_env import MultiAgentEnv
from .scenarios import load

import gymnasium as gym

class MPEPymarl(gym.Env):
    def __init__(self, batch_size=None, **kwargs):
        
        id = "simple_spread"
        scenario = load(id + ".py").Scenario()
        # create world
        self.episode_limit = 25
        
        render_mode = None
        world = scenario.make_world(
            render_mode=render_mode,
            world_length=self.episode_limit,
        )
        # create multiagent environment
        self.env = MultiAgentEnv(
            world,
            scenario.reset_world,
            scenario.reward,
            scenario.observation,
            scenario.info,
            render_mode=render_mode,
        )
        
        # Define the agents
        self.n_agents = self.env.n
        

        # Define the internal state
        self.n_actions = self.env.action_space.n
        
        self.obs_dim = self.env.observation_space["policy"].shape[0]
        self.state_dim = self.env.observation_space["critic"].shape[0]
        
        self.obs = np.zeros((self.n_agents, self.obs_dim))
        self.state = np.zeros((1, self.state_dim))
    
    def reset(self):
        """ Returns initial observations and states"""
        result, _ = self.env.reset()
        self.obs = result["policy"]
        self.state = result["critic"][:1]
        return self.obs, self.state

    def step(self, actions):
        """ Returns reward, terminated, info """
        result, reward, terminated, info = self.env.step(actions)
        self.obs = result["policy"]
        self.state = result["critic"][:1]
        return reward[0][0], terminated[0], info[0]

    def get_obs(self):
        """ Returns all agent observations in a list """
        return self.obs.copy()

    def get_state(self):
        return self.state.copy()

    def get_avail_actions(self):
        return np.ones([self.n_agents, self.n_actions])
    
    def get_obs_size(self):
        """ Returns the shape of the observation """
        return self.obs_dim
    
    def get_state_size(self):
        """ Returns the shape of the state"""
        return self.state_dim

    def get_total_actions(self):
        """ Returns the total number of actions an agent could ever take """
        return self.n_actions

    def get_env_info(self):
        env_info = {
            "state_shape": self.get_state_size(),
            "obs_shape": self.get_obs_size(),
            "n_actions": self.get_total_actions(),
            "n_agents": self.n_agents,
            "episode_limit": self.episode_limit
        }
        return env_info

    def get_stats(self):
        return None
        
        
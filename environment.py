# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 12:35:51 2023

@author: mahfouz
"""
#Imports
import functools
import numpy as np

import gymnasium
from gymnasium import spaces
from gymnasium.spaces import Discrete, Box
from pettingzoo import AECEnv
from pettingzoo.utils import agent_selector

MOVES = ["Move", "Not_Move"]

class raw_env(AECEnv):
    
    metadata = {"render_modes": ["human"], "name": "rps_v2"}

    def __init__(self, game, render_mode=None):
        
        super().__init__() 
        
        # Initial Parameters
        self.game_object = game
        self.NUM_ITERS = game.get_max_steps() 
        self.render_mode = render_mode
        self.num_moves = 0
        self.initial_resources = game.get_resources_players()
        self.resources = game.get_resources_players()
        # Agents
        self.agents = game.get_players()
        self.possible_agents = self.agents.copy()
        self.n_agents = game.get_num_players()
        self.agents_good = game.get_players_strategy_type("GOOD")
        self.agents_bad = game.get_players_strategy_type("BAD")
        self.n_agents_good = game.get_num_players_strategy_type("GOOD")
        self.n_agents_bad  = game.get_num_players_strategy_type("BAD")
        # optional: a mapping between agent name and ID
        self.agent_name_mapping = dict(zip(self.possible_agents, list(range(self.n_agents))))
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
        
        # Spaces
        # optional: we can define the observation and action spaces here as attributes to be used in their corresponding methods
        self.action_spaces = {agent: Discrete(agent.get_num_valid_actions()) for agent in self.possible_agents}
        self.observation_spaces = {agent: Discrete(agent.get_system().get_num_states()) for agent in self.possible_agents}
        # Define your environment-specific parameters and initialization here
        self.num_states =  {agent: agent.get_system().get_num_states() for agent in self.possible_agents}
        self.num_actions = {agent: agent.get_num_valid_actions() for agent in self.possible_agents}
        self.old_state_desirabilities = {}
        self.new_state_desirabilities = {}
        self.reward_devisor = {agent : agent.get_system().get_state_desirability_extrma()["max"] - agent.get_system().get_state_desirability_extrma()["min"] for agent in self.possible_agents}
    def seed(self, seed=None):
        np.random.seed(seed)

    def reset(self, seed=None, return_info=False, options=None):
        """
        Reset needs to initialize the following attributes
        - agents
        - rewards
        - _cumulative_rewards
        - terminations
        - truncations
        - infos
        - agent_selection
        And must set up the environment so that render(), step(), and observe()
        can be called without issues.
        Here it sets up the state dictionary which is used by step() and the observations dictionary which is used by step() and observe()
        """
        self.has_reset = True
        self.game_object.reset_game()
        self.agents = self.possible_agents[:]
        self.resources = self.initial_resources
        self.rewards = {agent: 0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {} for agent in self.agents}
        self.state = {agent: self.game_object.get_initial_state() for agent in self.agents}
        self.observations = {agent: agent.get_system().get_current_state() for agent in self.agents}
        self.num_moves = 0
        """
        Our agent_selector utility allows easy cyclic stepping through the agents list.
        """
        self._agent_selector = agent_selector(self.agents)
        self.agent_selection = self._agent_selector.next()
                
        # return self.observations, self.infos
        
    def observe(self, agent):
        """
        Observe should return the observation of the specified agent. This function
        should return a sane observation (though not necessarily the most up to date possible)
        at any time after reset() is called.
        """
        # observation of one agent is the previous state of the other
        return np.array(self.observations[agent])

    def step(self, action):
        """
        step(action) takes in an action for the current agent (specified by
        agent_selection) and needs to update
        - rewards
        - _cumulative_rewards (accumulating the rewards)
        - terminations
        - truncations
        - infos
        - agent_selection (to the next agent)
        And any internal state used by observe() or render()
        """
        # Removes terminated or truncated agents from 'everything'.
        if (self.terminations[self.agent_selection] or self.truncations[self.agent_selection]):
            # handles stepping an agent which is already dead
            # accepts a None action for the one agent, and moves the agent_selection to
            # the next dead agent,  or if there are no more dead agents, to the next live agent
            self._was_dead_step(action)
            return
        # Selects agent for this step
        agent = self.agent_selection
        # the agent which stepped last had its _cumulative_rewards accounted for
        # (because it was returned by last()), so the _cumulative_rewards for this
        # agent should start again at 0
        self._cumulative_rewards[agent] = 0
        
        action, cost = self.game_object.action_masking(agent, action)    
        self.resources[agent] = self.resources[agent] - cost
        # print(self.resources[agent])
        
        # Perform the specified actions by all agents and calculate the new state, rewards, and whether the episode is done
        self.old_state_desirabilities[agent] = agent.get_system().get_state_desirability(agent.get_system().get_current_state())
        self.game_object.perform_action(agent.get_system(), action)
        self.game_object.set_current_state()
        self.new_state_desirabilities[agent] = agent.get_system().get_state_desirability(agent.get_system().get_current_state())
        

        # update observation of next agent
        next_agent = self.agents[1 - self.agent_name_mapping[agent]]
        self.observations[next_agent] = next_agent.get_system().get_current_state()
        if self.observations[agent] != self.observations[next_agent]:
           self.game_object.set_undesirableness(self.new_state_desirabilities[agent])
             
        if self._agent_selector.is_last():
            # rewards for all agents are placed in the .rewards dictionary
            for agent in self.agents:               
              self.rewards[agent] = self.reward_function(agent, self.old_state_desirabilities[agent], self.new_state_desirabilities[agent])
            self.num_moves += 1
            # The truncations dictionary must be updated for all players.
            self.truncations = {
                agent: self.num_moves >= self.NUM_ITERS or self.game_object.is_game_over() for agent in self.agents
            }

            # observe the current state
            # for i in self.agents:
            #     self.observations[i] = self.state[
            #         self.agents[1 - self.agent_name_mapping[i]]
            #     ]
            
            self.game_object.increase_steps()
            self.timestep = self.game_object.get_current_step()            
        else:
            # necessary so that observe() returns a reasonable observation at all times.
            # self.state[self.agents[1 - self.agent_name_mapping[agent]]] = self.game_object.get_current_state()
            # no rewards are allocated until both players give an action
            self._clear_rewards()
        # selects the next agent.
        self.agent_selection = self._agent_selector.next()
        # Adds .rewards to ._cumulative_rewards
        self._accumulate_rewards()

        if self.render_mode == "human":
            self.render()
            
        # return self.observations, self.rewards, self.terminations, self.truncations,  self.infos

    def render(self, mode='human'):
        """
        Renders the environment. In human mode, it can print to terminal, open
        up a graphical window, or open up some other display that a human can see and understand.
        """
        if self.render_mode is None:
            gymnasium.logger.warn(
                "You are calling render method without specifying any render mode."
            )
            return

        if len(self.agents) == 2:
            string = "Current state: Agent1: {} , Agent2: {}".format(
                MOVES[self.state[self.agents[0]]], MOVES[self.state[self.agents[1]]]
            )
        else:
            string = "Game over"
        print(string)

    def close(self):
        # Implement the close method to clean up any resources or connections (optional)
        pass
    
    def reward_function(self, player, old_state_desirabilities, new_state_desirabilities):
        state_desirabilities_diff = np.array(old_state_desirabilities)-np.array(new_state_desirabilities)
        state_desirabilities_diff = player.get_player_strategy() * np.array(state_desirabilities_diff)
        reward = 0
        reward += state_desirabilities_diff#/ self.reward_devisor[player]
        player.set_score(reward)
        return reward
    
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent):
        return self.observation_spaces[agent]

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent):
        return self.action_spaces[agent]


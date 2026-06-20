# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 11:12:38 2023

@author: mahfouz
"""
#Imports
import random # importing the random module
import numpy as np
import csv
import os

from batchify import batchify, batchify_obs, unbatchify
VALID_ACTIONS = ["Not_Move", "Move"]

class Game:
    def __init__(self, System_Objects, max_steps, undesirableness_threashold):
        self.system_objects = System_Objects
        self.max_num_players = 2 * len(System_Objects)
        self.players = []
        self.players_good = []
        self.players_bad  = []
        self.initial_state = self.set_initial_state()
        self.current_state = self.initial_state
        self.current_step  = 0
        self.max_steps = max_steps
        self.game_over_flag = False 
        self.num_states = self.get_num_states()
        self.failure_states = self.set_failure_states()
        self.initial_resources = {}
        self.resources = {}
        self.cost_factor = 1
        self.players_good_score = 0
        self.players_bad_score = 0
        self.undesirableness = 0
        self.undesirableness_threashold = undesirableness_threashold

    
    def set_initial_state(self):
        initial_states = {}
        for system_object in self.system_objects:
            initial_states[system_object] = system_object.get_current_state()
        return initial_states
        
    def set_current_state(self):
        new_states = {}
        for system_object in self.system_objects:
            new_states[system_object] = system_object.get_current_state()
        self.current_state = new_states
        
    def set_failure_states(self):
        failure_states = {}
        for system_object in self.system_objects:
            failure_states[system_object] = []
            for obj in system_object.objects:
                if obj.element_type == "State":
                    if obj.state_desirability == '0':
                        failure_states[system_object].append(obj.state_name)
        return failure_states
    
    def set_score_team(self, points, team):
        if team == "GOOD":
           self.players_good_score+= points
        elif team == "BAD":
           self.players_bad_score+= points
        else:
           raise ValueError("There is no player with such strategy type")
    
    def set_undesirableness(self, state_desirabilities):
        self.undesirableness+= state_desirabilities
    
    def update_resources(self):
        for player in self.players:
            self.resources[player] = player.get_resources()
            
    def create_player(self, name, strategy_type, resources):
        if strategy_type == "GOOD":
            if len(self.players_good) < len(self.system_objects):
               player = Player(name, strategy_type, resources, self.system_objects[len(self.players_good)]) 
               self.initial_resources[player] = resources
               self.players_good.append(player)
               self.players.append(player)
               return player
            else:
              raise ValueError("Excedded Number of good team players") 
        elif strategy_type == "BAD":
            if len(self.players_bad) < len(self.system_objects):
               player = Player(name, strategy_type, resources, self.system_objects[len(self.players_bad)]) 
               self.initial_resources[player] = resources
               self.players_bad.append(player)
               self.players.append(player)
               return player
            else:
               raise ValueError("Excedded Number of bad team players") 
        else:
               raise ValueError("There is no player with such strategy type") 
               
    def get_valid_actions(self, player):
        return player.get_valid_actions()
    
    def is_game_over(self):
        if self.current_step == self.max_steps:
           self.game_over_flag = True
        elif all(state_type == "absorbing" for state_type in self.get_current_state_types()) and self.current_step > 1:
            self.game_over_flag = True
        elif self.undesirableness >= self.undesirableness_threashold:
              self.game_over_flag = True
        return self.game_over_flag
          
    def get_winner(self):
        if self.undesirableness >= self.undesirableness_threashold:
            winner_team = "BAD"
            return self.players_bad, winner_team
        scorers = []
        for i in range(len(self.players_good)):
            if self.players_good[i].get_score() > self.players_bad[i].get_score():
               scorers.append(self.players_good[i])
            elif self.players_good[i].get_score() < self.players_bad[i].get_score():
               scorers.append(self.players_bad[i]) 
        if self.players_good_score > self.players_bad_score:
           winner_team = "GOOD"
        elif self.players_good_score < self.players_bad_score:
           winner_team = "BAD"
        else:
           winner_team = "DRAW" 
        return scorers, winner_team
    
    def get_player_by_name(self, name):
        for player in self.players:
            if player.get_name() == name:
                return player
        raise ValueError("There is no player with such name")
                 
    def get_players(self):
        return self.players
    
    def get_players_strategy_type(self, strategy_type):
        if strategy_type == "GOOD":
           return self.players_good
        elif strategy_type == "BAD":
           return self.players_bad
        raise ValueError("There is no player with such strategy type")
    
    def get_num_players(self):
        return len(self.players)
    
    def get_num_players_strategy_type(self, strategy_type):
        if strategy_type == "GOOD":
           return len(self.players_good)
        elif strategy_type == "BAD":
           return len(self.players_bad)
        raise ValueError("There is no player with such strategy type")
        
    def get_current_state(self):
        current_states = []
        for system_object in self.system_objects:
            current_states.append(system_object.get_current_state())
        return current_states
    
    def get_current_state_desirabilities(self):
        state_desirabilities = []
        for system_object in self.system_objects:
            state_desirabilities.append(system_object.get_state_desirability(system_object.get_current_state()))
        return state_desirabilities
    
    def get_num_states(self):
        num_states = 1
        for system_object in self.system_objects:
            num_states = num_states * system_object.get_num_states()
        self.num_states = num_states
        return self.num_states
    
    def get_initial_state(self):
        return self.initial_state
    
    def get_game_over_flag(self):
        return self.game_over_flag
    
    def get_current_step(self):
        return self.current_step
    
    def get_max_steps(self):
        return self.max_steps
    
    def get_num_system_objects(self):
        return len(self.system_objects)
    
    def get_system_objects(self):
        return self.system_objects
    
    def get_current_state_names(self):
        current_state_names = []
        for system_object in self.system_objects:
            current_state_names.append(system_object.get_state_name(system_object.get_current_state()))
        return current_state_names
    
    def get_current_state_types(self):
        current_state_types = []
        for system_object in self.system_objects:
            current_state_types.append(system_object.get_state_type(system_object.get_current_state()))
        return current_state_types
        
    def get_resources_players(self):
        resources_players = {}
        for player in self.players:
            resources_players[player] = player.get_resources()
        return resources_players
    
    def get_score_team(self, team):
        if team == "GOOD":
           return self.players_good_score
        elif team == "BAD":
           return self.players_bad_score
        else:
           raise ValueError("There is no player with such strategy type")
           
    def get_undesirableness(self):
        return self.undesirableness
    
    def increase_steps(self):
        self.current_step  = self.current_step + 1
        
    def perform_action(self, system, action):
        if action == None:
            pass
        else:
            system.transition_to_state(action)
    
    def play(self, PATH, good_ai, bad_ai, rand_good_ai=False, rand_bad_ai=False):
        self.current_step = 0
        system_states = {}
        for system in self.system_objects:
            system_states[system] = []
            system_states[system].append("start")
            system_states[system].append(self.initial_state[system])

        PATH = PATH + 'CSV/player_behavior.csv'
        while (not self.game_over_flag):
            for player in self.players:
                old_state_desirabilities = player.get_system().get_state_desirability(player.get_system().get_current_state())
                # action = player.choose_action(self.get_valid_actions(player))
                observation = player.get_system().get_current_state()
                observation_names =self.get_current_state_names()
                if player.get_player_strategy() == 1:
                    model = good_ai
                else:
                    model = bad_ai
                obs = batchify_obs(observation, player, model.device)
                player.update_valid_actions_mask(player.get_system())
                if player.get_player_strategy() == 1: 
                   if not rand_good_ai:
                      action_mask = batchify({player:player.get_valid_actions_mask()},  model.device)
                      action,_,_,_ = model.get_action_and_value(obs, player, invalid_action_masks = action_mask )
                      player_action = unbatchify(action)[0]                      
                   else:
                      action_mask = player.get_valid_actions_mask()
                      player_action = model.get_random_action(action_mask)
                else: 
                   if not rand_bad_ai:
                      action_mask = batchify({player:player.get_valid_actions_mask()},  model.device)
                      action,_,_,_ = model.get_action_and_value(obs, player, invalid_action_masks = action_mask )
                      player_action = unbatchify(action)[0]                      
                   else:
                      action_mask = player.get_valid_actions_mask()
                      player_action = model.get_random_action(action_mask)
                player_action, _ = self.action_masking(player, player_action)       
                self.perform_action( player.get_system(), player_action)
                self.set_current_state()
                if self.initial_state[player.get_system()] == player.get_system().get_current_state():
                   system_states[player.get_system()].append("start")
                system_states[player.get_system()].append(player.get_system().get_current_state())
                new_state_desirabilities = player.get_system().get_state_desirability(player.get_system().get_current_state())
                state_desirabilities_diff = np.array(old_state_desirabilities)-np.array(new_state_desirabilities)
                state_desirabilities_diff = player.get_player_strategy() * np.array(state_desirabilities_diff)
                player.set_score(state_desirabilities_diff)
                if observation != player.get_system().get_current_state():
                   self.set_undesirableness(new_state_desirabilities)
                self.set_score_team(state_desirabilities_diff, player.get_player_strategy_name())
                self.record_player_behavior(PATH, player, observation_names, player_action)
            self.increase_steps()
            self.is_game_over()
        scorers, winner_team = self.get_winner()
        paths_ocurrs_systems = {}
        for system in self.system_objects:
            print(system_states[system])
            print("-------------------------")
            paths = system.paths
            # print(paths)
            paths_ocurrs = {}
            for i in range(len(paths)):
                paths_ocurrs[f"Path_{i}"] = 0 
            unique_cycles = self.split_array_on_item(system_states[system], "start")
            for unique_cycle in unique_cycles: 
                unique_states = self.remove_duplicates(unique_cycle)
                # print(unique_states)
                for i in range(len(paths)):
                    if unique_states == paths[i]:
                        paths_ocurrs[f"Path_{i}"]+= 1 
                # print("************************")
            # print(paths_ocurrs)
            paths_ocurrs_systems[system] = paths_ocurrs
        for player in self.players:
            print(player.get_name())
            print(player.get_score())
        if winner_team == "GOOD":
            print("Good wins in " + str(self.current_step) + " rounds.")
        elif winner_team == "BAD":
            print ("Bad wins in " + str(self.current_step) + " rounds.")
        else:
            print ("End with in " + str(self.current_step) + " rounds.") 
        final_state = []
        print(f"Cummulative undesirableness = {self.undesirableness}")
        for i in range(len(self.system_objects)):
          final_state.append(self.system_objects[i].get_state_name(self.system_objects[i].get_current_state()))
        print(final_state)
        if (self.check_system_faliure()):
            print("BAD AI won regardless the game score")
        return scorers, winner_team, paths_ocurrs_systems
                   
            
    def action_masking(self, player, action):
        # Action Mask - Poor implementation?, they've done it differently in the PettingZoo Chess documentation where a mask is returned along with the observations in the observe function.  
        mask = player.get_valid_actions_mask()
        system = player.get_system()
        if mask[action] == 0: # If Action impossible, do no action. Might want to count illegal actions here and add to information.
            action = None
            cost = 0
            player.set_took_invalid_action(True)
        else: # Else check if I can afford action.
            trans_prop = system.get_transition_matrix()[system.get_current_state()][action]
            cost = self.cost_factor / trans_prop
            if cost > self.resources[player]: # If I cannot afford action, do no action.
                action = None
                cost = 0
                player.set_took_invalid_action(True)
            else:
                player.set_took_invalid_action(False)    
        self.resources[player] = self.resources[player] - cost
        player.set_resources(self.resources[player])
        
        return action, cost
        
    def reset_game(self):
        for system_object in self.system_objects:
          system_object.reset_system()
          
        for player in self.players:
           player.reset_player()
           self.resources[player] = self.initial_resources[player]
               
        self.current_state = self.initial_state
        self.current_step  = 0  
        self.game_over_flag = False
        self.players_good_score = 0
        self.players_bad_score = 0
        self.undesirableness = 0
    
    def record_player_behavior(self, path,  player, observation, action_mask): 
      if action_mask != None:
         action = player.get_valid_actions()[action_mask]
      else:
          action = None
      header = ['timestamp', 'player_name', 'player_obs', 'player_res', 'player_action', 'next_state', 'player_score']    
      data   = [self.current_step, player.get_name(), observation, player.get_resources(), action, self.get_current_state_names(), player.get_score()]
      isExist=os.path.exists(path)
      if (not isExist):
         f = open(path, 'a+', newline='', encoding='utf-8')
         writer = csv.writer(f)
         writer.writerow(header)
         writer.writerow(data) 
      else:
         f = open(path, 'a+', newline='', encoding='utf-8')
         writer = csv.writer(f)
         writer.writerow(data) 
      
    def check_system_faliure(self):
        current_state_names = self.get_current_state_names()
        for state in current_state_names:
            for system_object in self.system_objects:
                if state in self.failure_states[system_object]:
                    return True
        return False
    
    def remove_duplicates(self, arr):
        seen = set()
        result = []
    
        for item in arr:
            if item not in seen:
                seen.add(item)
                result.append(item)
    
        return result
    
    def split_array_on_item(self, arr, item_to_split_on):
       subarrays = []
       current_subarray = []

       for item in arr:
           if item == item_to_split_on:
               if current_subarray:
                   subarrays.append(current_subarray)
               current_subarray = []
           else:
               current_subarray.append(item)

       if current_subarray:
           subarrays.append(current_subarray)

       return subarrays
        
class Player:
    def __init__(self, name, strategy_type, resources, system):
        self.name = name
        self.masks = {"Not_Active" : 0, "Active": 1}
        self.system = system
        self.valid_actions, self.valid_actions_mask  = self.create_valid_actions(system)
        self.valid_actions_costs = []
        self.score = 0
        self.strategy_type  = strategy_type
        self.initial_resources = resources
        self.resources = self.initial_resources
        self.took_invalid_action = False
        self.action_invalid_states = {}
        
    def choose_action(self, valid_actions):
        return valid_actions[random.randint(0,len(valid_actions) - 1)]

    def add_valid_action(self, action, cost, action_invalid_states):
        if action in VALID_ACTIONS:
            self.valid_actions.append(action)
            self.valid_actions_mask.append(self.masks["Active"])
            self.valid_actions_costs.append(cost)
            self.action_invalid_states[action] = action_invalid_states
        else:
            raise ValueError("Not valid action") 
              
    def create_valid_actions(self, system):
        valid_actions = []
        valid_actions_mask = []
        max_num_states = system.get_num_states()
        for i in range(max_num_states):
            valid_actions.append(i)
            valid_actions_mask.append(self.masks["Active"])
        return valid_actions, valid_actions_mask
        
    def increase_score(self):
        self.score = self.score + 1
    
    def decrease_score(self):
        self.score = self.score - 1
        
    def set_score(self, points):
        self.score = self.score + points
        

    def get_valid_actions(self): 
        return self.valid_actions
    
    def get_valid_actions_mask(self): 
        return self.valid_actions_mask
      
    def get_action_num(self, action):
        for i in range(len(self.get_valid_actions())):
            if action == self.get_valid_actions()[i]:
                return i
        raise ValueError("There is no such action name")
                
    def get_num_valid_actions(self): 
        return len(self.valid_actions)
    
    def get_score(self):
        return self.score
    
    def get_name(self):
        return self.name
    
    def get_player_strategy(self):
        if self.strategy_type == 'GOOD':
            return 1
        elif self.strategy_type == 'BAD':
            return -1
        return 0
    
    def get_player_strategy_name(self):
        return self.strategy_type
      
    def get_resources(self):
        return self.resources
    
    def set_resources(self, resources):
        self.resources = resources
    
    def get_action_cost(self, action):
        return self.valid_actions_costs[action]
        
    def get_took_invalid_action(self):
        return self.took_invalid_action
        
    def set_action_cost(self, action, cost):
        self.valid_actions_costs[action] = cost
        
    def get_action_invalid_states(self):
        return self.action_invalid_states
        
    def get_system(self):
        return self.system
    
    def update_valid_actions_mask(self, system): 
        possible_next_states = system.get_possible_next_state()
        for i in range(len(self.valid_actions)):
            if i in possible_next_states:
                self.valid_actions_mask[i] = self.masks["Active"]
            else: 
                self.valid_actions_mask[i] = self.masks["Not_Active"]

    def set_took_invalid_action(self,value):
        self.took_invalid_action = value
        
    def reset_player(self):
        self.score = 0
        self.resources = self.initial_resources
     
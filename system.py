'''
title		:	Dynamic Fault Tree Game
version     :   0.0.1
date		:	27.03.2023
fileName	:	system.py
author		:	Joachim Nilsen Grimstad
contact 	:   Joachim.Grimstad@ias.uni-stuttgart.de
edit        :   on Wed Jun 21 12:05:54 2023 by Ahmed Mahfouz 
description :   System
license 	:   This tool is licensed under Creative commons Attribution-NonCommercial-ShareAlike 4.0 International (CC BY-NC-SA 4.0).
                For license details, see https://creativecommons.org/licenses/by-nc-sa/4.0/  
disclaimer	:   Author takes no responsibility for any use.
'''

# Imports
from dft import No_Action
import numpy as np   
# Classes

class System:
    'System Model Objects'
    def __init__(self, name, type):
        self.name = name
        self.type = type
        self.objects = []
        self.clock = 0

    def get_object(self, object_name):
        'returns the object with a given object_name'
        for obj in self.objects:
            if obj.name == object_name:
                return obj
    def get_name(self):
        return self.name

class System_DFT(System):
    'Dynamic Fault Tree System Model Object'
    def __init__(self, name):
        System.__init__(self, name, 'DFT')

    # ___________________________Methods_______________________________
    
    def reset_system(self):
        'resets system'
        self.clock = 0
        for obj in self.objects:
            obj.reset()
        self.top_event.update_state()
        self.state = self.top_event.state

    def instantiate(self):
        'instantiates the system, updates the states of all the events based on basic events.'
        self.top_event = self.get_top_event()
        self.reset_system()
        self._set_actions()
        self._set_observations()
        
    def get_top_event(self):
        'returs the top event'
        for obj in self.objects:
            if obj.type == 'TOP':
                return obj

    def _set_actions(self):
        no_action = No_Action('No Action')
        actions = []
        actions.append(no_action)
        for obj in self.objects:
            if obj.type =='BASIC':
                actions.append(obj)
        self.actions = actions

    def _set_observations(self):
        observations = []
        for obj in self.objects:
            if obj.type == 'BASIC':
                observations.append(obj)
        self.observations = observations

    def apply_action(self, agent, action):
        if agent == 'agent_red':
            self.actions[action].red_action()
        elif agent == 'agent_blue':
            self.actions[action].blue_action()

    def time_step(self):
        for obj in self.observations:
            obj.time_step()
        self.update_system()
        self.clock += 1 
        
    def update_system(self):
        self.top_event.update_state()
        self.state = self.top_event.state

class System_DTMC(System):
    'Descrete-Time Markov Chain System Model Object'
    def __init__(self, name):
        System.__init__(self, name, 'DTMC')
        self.transition_matrix = []
        self.num_states    = 0
        self.num_edges     = 0
        self.initial_state = None
        self.end_state = None
        self.current_state = self.initial_state
        self.transitions = {}  # Dictionary to store transitions
 # ___________________________Methods_______________________________  
    
    def reset_system(self):
        'resets system'
        self.clock = 0
        for obj in self.objects:
            obj.reset()
        self.update_transition_matrix_from_objects(self.objects)
        self.current_state = self.initial_state

    def update_transition_matrix_from_objects(self, objects):
        num_states = 0
        num_edges  = 0 
        for obj in objects:
            if obj.element_type == 'State':
                num_states = num_states + 1
            else: 
                num_edges = num_edges + 1
        self.set_num_states(num_states)
        self.set_num_edges(num_edges)
        transition_matrix  = np.zeros((num_states,num_states))
        for obj in objects:
            if obj.element_type == 'Edge':
                transition_matrix[obj.source_state][obj.target_state] = obj.prob_value        
        if (self._is_valid_transition_matrix(transition_matrix)):
            self.set_transition_matrix(transition_matrix)
            for obj in objects:
                if obj.element_type == 'Edge':
                   self.add_transition(obj.source_state,obj.target_state,obj.prob_value)   
        else:
            print(transition_matrix)
            raise ValueError("Invalid transition matrix") 
    
    def set_num_states(self, num_states):
        self.num_states = num_states
        
    def set_num_edges(self, num_edges):
        self.num_edges = num_edges
        
    def set_current_state(self, current_state):
        self.current_state = current_state
            
    def set_initial_state(self, state):
        if state < 0 or state >= self.num_states:
            raise ValueError("Invalid initial state")
        self.initial_state = state
        self.current_state = state
        
    def set_end_state(self, state):
        if state < 0 or state >= self.num_states:
            raise ValueError("Invalid initial state")
        self.end_state = state
        self.paths = self.find_paths(self.initial_state, self.end_state)
   
    def set_state(self, state):
        self.current_state = state
        
    def set_transition_matrix(self, transition_matrix):
        self.transition_matrix = np.array(transition_matrix)
    
    def transition_to_state(self, state):
        if self.current_state is None:
            raise ValueError("Initial state not set")
        if state < 0 or state >= self.num_states:
            raise ValueError("Invalid target state")
        transition_prob = self.transition_matrix[self.current_state][state]
        if np.random.rand() < transition_prob:
            self.set_current_state(state)
            
    def transition(self):
        probabilities = self.transition_matrix[self.current_state]
        next_state = np.random.choice(range(len(probabilities)), p=probabilities)
        self.set_current_state(next_state)
        return next_state
            
    def _is_valid_transition_matrix(self, transition_matrix):
        # Check if transition matrix is square
        if transition_matrix.shape[0] != transition_matrix.shape[1]:
            return False
        
        # Check if transition probabilities are valid
        for row in transition_matrix:
            if np.any(row < 0) or np.abs(np.sum(row) - 1) > 1e-6:
                return False
        
        return True
    
    def get_current_state(self):
        return self.current_state
         
    def get_next_state(self):
        transition_probs = self.transition_matrix[self.current_state]
        next_state = np.random.choice(len(transition_probs), p=transition_probs)
        # self.current_state = next_state
        return next_state
    
    def get_possible_next_state(self):
        transition_probs = self.transition_matrix[self.current_state]
        possible_next_states = []
        for i in range(len(transition_probs)):
            if transition_probs[i] > 0:
               possible_next_states.append(i)        
        return possible_next_states
  
    def get_transition_matrix(self):
        return self.transition_matrix
    
    def get_num_states(self):
        return self.num_states 
    
    def get_state_desirability(self, state):
        for obj in self.objects:
          if obj.element_type == "State":
            if obj.state_number == state:
                return obj.state_desirability
        raise ValueError("There is no such state")
        
    def get_state_desirability_extrma(self):
      max_des = -1000
      min_des =  1000
      for obj in self.objects:
          if obj.element_type == "State":
              if obj.state_desirability > max_des:
                 max_des = obj.state_desirability
              elif obj.state_desirability < min_des:
                  min_des = obj.state_desirability
      return {"max":max_des, "min":min_des}
        
    
    def get_state_name(self, state):
        for obj in self.objects:
          if obj.element_type == "State":
            if obj.state_number == state:
                return obj.state_name
        raise ValueError("There is no such state")
        
    def get_state_type(self, state):
        for obj in self.objects:
          if obj.element_type == "State":
            if obj.state_number == state:
                return obj.state_type
        raise ValueError("There is no such state")
        
    def get_state_number(self, name):
        for obj in self.objects:
          if obj.element_type == "State":
            if obj.state_name == name:
                return obj.state_number
        raise ValueError("There is no such state")
        
        
    def simulate(self, num_steps):
        if self.current_state is None:
            raise ValueError("Initial state not set")

        states = [self.current_state]
        for _ in range(num_steps):
            probabilities = self.transition_matrix[self.current_state]
            probabilities /= probabilities.sum()  # normalize
            next_state = np.random.choice(self.num_states, p=probabilities)
            states.append(next_state)
            self.set_current_state(next_state)

        return states
    
    def add_transition(self, from_state, to_state, probability):
        if from_state not in self.transitions:
            self.transitions[from_state] = []
        self.transitions[from_state].append((to_state, probability))
        
   
    def find_paths(self, start_state, end_state):
        def dfs(current_state, path):
            if current_state == end_state:
                paths.append(path[:])  # Append a copy of the path
                return
            if current_state not in self.transitions:
                return
            for next_state, probability in self.transitions[current_state]:
                if next_state not in path:
                    path.append(next_state)
                    dfs(next_state, path)
                    path.pop()

        paths = []
        dfs(start_state, [start_state])
        return paths
        
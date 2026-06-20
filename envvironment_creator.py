# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 12:07:22 2023

@author: mahfouz
"""

#Imports
from parsing import Parse
from game import Game
from environment import raw_env

AGENTS_TYPE = {0:"GOOD", 1:"BAD", 2:"OPPONENTS"}
    
def env_creator(model_files, agent_type, state_init, state_end):
    Parser = Parse(name="Parser")
    systems = []
    n = len(model_files)
    for i in range(n): 
        system = Parser.from_file(model_files[i])
        system.reset_system()
        system.set_initial_state(state_init[i])
        system.set_end_state(state_end[i])
        systems.append(system)

    game = Game(System_Objects=systems, max_steps=25, undesirableness_threashold=15)
        
    if agent_type == AGENTS_TYPE[0]:
       game.create_player("agent_blue", "GOOD", 1000)  
       game.create_player("agent_black", "GOOD", 1000)
    
    elif agent_type == AGENTS_TYPE[1]:
       game.create_player("agent_red", "BAD", 1000)
       game.create_player("agent_white", "BAD", 1000)
       
    elif agent_type == AGENTS_TYPE[2]:
       for i in range(game.get_num_system_objects()):
           game.create_player("agent_red_" + str(i), "BAD", 1000)
           game.create_player("agent_blue_" + str(i), "GOOD", 1000)  

    
    env = raw_env(game)
   
    return env, game


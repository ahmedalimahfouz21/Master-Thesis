# -*- coding: utf-8 -*-
"""
Created on Wed Jun 21 19:29:08 2023

@author: mahfouz
"""
#Imports
from ppo import train, create_ppo_agent
from envvironment_creator import env_creator
from analysis import analize_game_behavour_file, experiment_report

INPUT_MODELS = {0:'drunkardswalk.xml', 1:'examplerobot.xml', 2:'cleaningrobot.xml', 3:'onlineshopping.xml', 4:"httprequest.xml", 5:'teleassistance.xml', 6:'travelassistance.xml', 7:'RandomModel.xml', 8:"SimpleModel.xml"}
AGENTS_TYPE = {0:"GOOD", 1:"BAD", 2:"OPPONENTS"}

def Training(PATH, env, total_episodes):

 '''Training'''
 if total_episodes == 0:
     return
 
 actor_model = PATH + '/ppo_actor_'
 critic_model =  PATH + '/ppo_critic_'
 train(PATH, env, actor_model, critic_model, total_episodes = total_episodes)
 
def Simulating(PATH_SOURCE, PATH_SINK, env, game, simulations):
     
 '''Simulating'''
 if simulations == 0:
     return
 
 good_ai = create_ppo_agent(env, PATH_SOURCE + '/ppo_actor_',  PATH_SOURCE + '/ppo_critic_')
 bad_ai  = create_ppo_agent(env, PATH_SOURCE + '/ppo_actor_',  PATH_SOURCE + '/ppo_critic_')
 GOOD_TEAM_WINS, BAD_TEAM_WINS, END_WITH_DRAW  = 0, 0, 0
 paths_ocurrs_games = {}
 for system in game.get_system_objects():
     paths_ocurrs_games[system] = {}
     for i in range (len(system.paths)):
          paths_ocurrs_games[system][f"Path_{i}"] = 0
             
 for i in range(simulations):
     game.reset_game()
     print("Simulation for " + game.get_system_objects()[0].get_name()+ ":" + str(i))
     rand_good_ai = False
     rand_bad_ai = False
     scorers, winner_team, paths_ocurrs_systems = game.play(PATH_SINK, good_ai=good_ai,bad_ai=bad_ai, rand_good_ai=rand_good_ai , rand_bad_ai=rand_bad_ai )
     if winner_team == "DRAW":
        END_WITH_DRAW+=1
     elif winner_team  == "GOOD":
          GOOD_TEAM_WINS+=1
     else:
          BAD_TEAM_WINS+=1
     for system in game.get_system_objects():
         for i in range (len(system.paths)):
            paths_ocurrs_games[system][f"Path_{i}"]+= paths_ocurrs_systems[system][f"Path_{i}"]
     print("\n----------------------------------------------------\n")
 describtion = Exp_description()
 experiment_report(game, PATH_SINK, describtion, simulations, paths_ocurrs_games, GOOD_TEAM_WINS, BAD_TEAM_WINS, END_WITH_DRAW, rand_good_ai=rand_good_ai , rand_bad_ai=rand_bad_ai )
 
def Analyzing(PATH, game):

  '''Analyzing'''
  
  analize_game_behavour_file(PATH, game)

def Exp_description():
    Describtion = "Description : Agents were trained with number of Epochs = 100000 \n"
    return Describtion

if __name__ == "__main__":
  device_folder = "."
  models = [7]
  initial_state = [0]
  end_state = [14]
  model_files = []
  for k in models:
     model_file = device_folder + '/DTMC_Models/' + INPUT_MODELS[k]   
     model_files.append(model_file)
     
  env,game = env_creator(model_files, AGENTS_TYPE[2], initial_state, end_state) 
  
  PATH_PPO = device_folder + '/PPO_Agent/' + game.get_system_objects()[0].get_name()
  Training(PATH_PPO, env, total_episodes = 0)
 
  env.reset()
  PATH_Records = device_folder + '/Game_Records/' + game.get_system_objects()[0].get_name() + '/' 
  Simulating(PATH_PPO, PATH_Records, env, game, simulations=100000)
  # Analyzing(PATH_Records, game)
  print("\n------------------------\n")
  print("Done")
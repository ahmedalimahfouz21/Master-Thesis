# -*- coding: utf-8 -*-
"""
Created on Sat Jul  8 13:44:34 2023

@author: mahfouz
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np  
import csv
import os
import datetime

Plot_figures = False
Record_obs_act = True

def analize_game_behavour_file(Source_path,Game):
    Data = pd.read_csv(Source_path + 'CSV/player_behavior.csv')
    timestamps = Data['timestamp']
    player_names = Data['player_name']
    player_obses = Data['player_obs']
    player_actions = Data['player_action']
    game_started = False
    game_num = 0
    i = 0
    while i < len(timestamps):
        if timestamps[i] == 0 and not game_started:
           agents = []
           observations = {}
           actions = {}
           obs_action_matrix  = {} 
           players_num=0
           while (timestamps[i + players_num] == 0):
               agents.append(player_names[i + players_num])
               observations[player_names[i + players_num]] = []
               actions[player_names[i + players_num]] = []
               obs_action_matrix[player_names[i + players_num]] = np.zeros((Game.get_num_states(),Game.get_players()[0].get_num_valid_actions()))
               players_num+=1
           game_started = True
           game_num+= 1
           # print(game_num)
        # print(agents)
        # print(i)
        if i < (len(timestamps) - players_num ):
          if (i + players_num) < len(timestamps):
             if timestamps[i + players_num] == 0:
               for k in range(players_num):
                   observations[player_names[i+k]].append(player_obses[i+k])
                   actions[player_names[i+k]].append(player_actions[i+k])
               game_started = False
               if Plot_figures:
                  plt.figure()
                  plt.grid()
                  plt.title('Running action/observation of ' + 'Game_num: ' + str(game_num))
                  Figures_path = Source_path + 'Figures/'
                  figure_file = Figures_path + 'Game_' + str(game_num) + '.png' 
                  for p in range(players_num):
                      x = [o for o in range(len(observations[player_names[i+p]]))]
                      plt.plot(x, observations[player_names[i+p]])
                      plt.savefig(figure_file)
                  for p in range(players_num):
                      x = [a for a in range(len(actions[player_names[i+p]]))]
                      plt.plot(x, actions[player_names[i+p]])   
                      plt.savefig(figure_file)
               for p in range(players_num):
                   for o in range(len(observations[player_names[i+p]])):
                       state_string = observations[player_names[i+p]][o]
                       state_number = Game.get_system_objects()[0].get_state_number(state_string[2:len(state_string) - 2])
                       action_number = Game.get_player_by_name(player_names[i+p]).get_action_num(actions[player_names[i+p]][o])
                       obs_action_matrix[player_names[i+p]][state_number][action_number]+= 1
                   if (Record_obs_act):
                       record_player_obs_act(Source_path, game_num, len(observations[player_names[i+p]]),player_names[i+p], obs_action_matrix[player_names[i+p]], Game)                       
               i = i + players_num
               # print(obs_action_matrix)
             else:
               observations[player_names[i]].append(player_obses[i])
               actions[player_names[i]].append(player_actions[i])
               i+=1
        else:
               for k in range(players_num):
                  observations[player_names[i+k]].append(player_obses[i+k])
                  actions[player_names[i+k]].append(player_actions[i+k])
               # print(i)
               game_started = False
               if Plot_figures:
                  plt.figure()
                  plt.grid()
                  plt.title('Running action/observation of ' + 'Game_num: ' + str(game_num))
                  figure_file = Figures_path + 'Game_' + str(game_num) + '.png' 
                  for p in range(players_num):
                      # plt.figure()
                      # plt.grid()
                      x = [o for o in range(len(observations[player_names[i+p]]))]
                      plt.plot(x, observations[player_names[i+p]])
                      # plt.plot(x, actions[player_names[p]])
                      # plt.title('Running action/observation of ' + agents[p] + ' Game_num : ' + str(game_num))
                      # figure_file = Figures_path + agents[p] + '_Game_' + str(game_num) + '.png'     
                      plt.savefig(figure_file)
                  for p in range(players_num):
                      x = [a for a in range(len(actions[player_names[i+p]]))]
                      # plt.plot(x, observations[player_names[p]])
                      plt.plot(x, actions[player_names[i+p]])   
                      plt.savefig(figure_file) 
               for p in range(players_num):
                   for o in range(len(observations[player_names[i+p]])):
                       state_string = observations[player_names[i+p]][o]
                       state_number = Game.get_system_objects()[0].get_state_number(state_string[2:len(state_string) - 2])
                       action_number = Game.get_player_by_name(player_names[i+p]).get_action_num(actions[player_names[i+p]][o])
                       obs_action_matrix[player_names[i+p]][state_number][action_number]+= 1
                   if (Record_obs_act):
                       record_player_obs_act(Source_path, game_num, len(observations[player_names[i+p]]), player_names[i+p], obs_action_matrix[player_names[i+p]], Game) 
               i = i + players_num               
               # print(obs_action_matrix)
              
        # print(observations)
        # print(actions)   
def record_player_obs_act(Source_path, game_num, game_rounds, player_name, obs_action_matrix, Game):
      data   = [game_num, game_rounds ,player_name, Game.get_player_by_name(player_name).get_valid_actions()]
      CSV_Path = Source_path + 'CSV/player_obs_act.csv'
      for i in range(len(obs_action_matrix)):
          data.append(obs_action_matrix[i])
      isExist=os.path.exists(CSV_Path)
      if (not isExist):
         header = ['game_num', 'game_rounds', 'player_name', 'Valid_actions']
         for n in range(Game.get_num_states()):
            state_name = Game.get_system_objects()[0].get_state_name(n)
            header.append(state_name)
         f = open(CSV_Path, 'a+', newline='', encoding='utf-8')
         writer = csv.writer(f)
         writer.writerow(header)
         writer.writerow(data) 
      else:
         f = open(CSV_Path, 'a+', newline='', encoding='utf-8')
         writer = csv.writer(f)
         writer.writerow(data) 
         
def experiment_report(Game, Path, Describtion, simulations, paths_ocurrs, GOOD_TEAM_WINS, BAD_TEAM_WINS, END_WITH_DRAW, rand_good_ai = False, rand_bad_ai = False):
    
     GOOD_TEAM_WINS_PERCENT = 100 * GOOD_TEAM_WINS/ simulations
     BAD_TEAM_WINS_PERCENT  = 100 * BAD_TEAM_WINS/ simulations
     END_WITH_DRAW_PERCENT  = 100 * END_WITH_DRAW/ simulations
     
     string = "Experiment Summary: "
     if rand_good_ai and rand_bad_ai:
        string = string + "Random_Vs_Random \n"
     elif not rand_good_ai and not rand_bad_ai: 
        string = string + "Blue_Vs_Red \n"
     elif not rand_good_ai and  rand_bad_ai: 
        string = string + "Blue_Vs_Random \n"
     else:
        string = string + "Random_Vs_Red \n" 
     string = string + Describtion
     string = string + f"Total number of game simulations = {simulations} \n"
     string = string + f"Good Team wins {GOOD_TEAM_WINS} times with {GOOD_TEAM_WINS_PERCENT} % \n"
     string = string + f"Bad Team wins {BAD_TEAM_WINS} times with {BAD_TEAM_WINS_PERCENT} % \n"
     string = string + f"End Team wins {END_WITH_DRAW} times with {END_WITH_DRAW_PERCENT} % \n"
     string = string + "Safety result: "
     if BAD_TEAM_WINS_PERCENT >= GOOD_TEAM_WINS_PERCENT:
         string = string + "System is unsafe \n"
     elif  GOOD_TEAM_WINS_PERCENT -  BAD_TEAM_WINS_PERCENT <= 10:
         string = string + "System is critacally safe \n"
     else:
         string = string + "System is safe \n"
     # PATH report
     string = string + "Paths Records: \n"
     for system in Game.get_system_objects():
         string = string + system.get_name() + "\n"
         for i in range(len(system.paths)):
             times = paths_ocurrs[system][f"Path_{i}"]
             string = string + f"Path_{i}: {system.paths[i]}" + " was taken " + f"{times}" + " times.\n"  
         
     # Your string to be saved
     text_to_save = string

     # Specify the file path
     file_name = "Ex_" + get_date_time()
     file_path = Path + "/Experiment/" + file_name + ".txt"  # Change this to the desired file path

     # Open the file in write mode and save the string
     with open(file_path, 'w') as file:
        file.write(text_to_save)
   
     print(text_to_save)
     print("String saved to", file_path)
     
def get_date_time():
    
    # Get the current date and time
    current_datetime = datetime.datetime.now()

    # Convert the datetime object to a string
    date_time_string = current_datetime.strftime("%Y_%m_%d_%H_%M_%S")

    # Print the date and time as a string
    return  date_time_string

        
    
# -*- coding: utf-8 -*-
"""
Created on Sat Aug 19 11:27:06 2023

@author: mahfouz
"""
# Imports
import numpy as np
import networkx as nx
import random
import matplotlib.pyplot as plt  


def generate_model(num_states, max_edges, model_name, desirable_states):
 
 transition_matrix  = generate_random_transition_matrix(num_states, max_edges)
 draw_transtion_matrix(transition_matrix, model_name)
 draw_heat_map(transition_matrix, model_name)
 # print(transition_matrix)
 string = "<?xml version=" + qoutes(str(1.0)) + "?>"
 string = string + "\n"
 string = string + "<model type=" + qoutes("DTMC") + " name=" + qoutes(model_name) + ">"
 string = string + "\n\n"
 for i in range(num_states):
     string = string + indentation() + "<define-state name=" + qoutes("S" + str(i))
     if i<10: 
        string = string + " "
     string = string + " " + "number=" + qoutes(str(i)) + " "
     if i<10: 
         string = string + " "
     string = string + "type="
     if transition_matrix[i][i] == 1.0:
         string = string + qoutes("absorbing")
     else:
         string = string + qoutes("transient")
     string = string + " state_undesirability = "
     if  i < desirable_states :   
         undesirability = np.random.randint(-num_states/2, 0)
     else:
         undesirability = np.random.randint(0, num_states/2)
     # undesirability = np.random.randint(-num_states/2, num_states/2)
     string = string + qoutes(str(undesirability)) + "/>"
     string = string + "\n"
     
 string = string + "\n"
     
 for i in range(num_states):
     for k in range(num_states):
         if transition_matrix[i][k] > 0:
            string = string + indentation() + "<define-edge from=" + qoutes(str(i)) + " to=" + qoutes(str(k)) 
            if k<10: 
               string = string + " "
            string = string +" value=" 
            string = string + qoutes(str(transition_matrix[i][k])) + "/>"
            string = string + "\n"
     string = string + "\n"
 
 string = string + "\n\n"
 string = string + "</model>"
 
 # Your string to be saved
 text_to_save = string

# Specify the file path
 file_path = "./DTMC_Models/" + model_name + ".xml"  # Change this to the desired file path

# Open the file in write mode and save the string
 with open(file_path, 'w') as file:
     file.write(text_to_save)
   
 print(string)
 print("String saved to", file_path)
 
 
 
def generate_random_transition_matrix(num_states, max_edges):
    # Generate a random matrix with elements between 0 and 1
    random_matrix = np.zeros((num_states,num_states))
    random_matrix[0][1] = 1.0
    random_matrix[num_states - 2][0] = 0.9
    random_matrix[num_states - 2][num_states - 1] = 0.1  
    random_matrix[num_states - 1][num_states - 1] = 1.0  # One Absorbing State
    for i in range(1, num_states - 2):
        num_random_state = np.random.randint(2, max_edges +1)
        random_states =[]
        while len(random_states) < num_random_state:
            random_state = np.random.randint(1, num_states - 1)
            if i != random_state:
               random_states.append(random_state)
        for random_state in random_states:
            random_matrix[i][random_state] = random.random()
            
    # Normalize each row to ensure they sum up to 1
    transition_matrix = random_matrix / random_matrix.sum(axis=1, keepdims=True)
    
    return transition_matrix
def draw_transtion_matrix(transition_matrix,model_name):
  file_path = "./DTMC_Models/" + model_name + "_dtmc.png"  # Change this to the desired file path
  num_states = len(transition_matrix)
    
  # Create a directed graph
  G = nx.DiGraph()
    
  # Add nodes
  for i in range(num_states):
      G.add_node(i)
    
  # Add edges based on transition probabilities
  for i in range(num_states):
      for j in range(num_states):
          probability = transition_matrix[i, j]
          if probability > 0:
              G.add_edge(i, j, weight=probability)    
  # Draw the graph
  pos = nx.spring_layout(G)
  labels = {(i, j): f"{G[i][j]['weight']:.2f}" for i, j in G.edges}
  nx.draw(G, pos, with_labels=True, node_size=1000, node_color="skyblue", font_size=10)
  nx.draw_networkx_edge_labels(G, pos, edge_labels=labels)
  
  plt.savefig(file_path)  
  plt.show()

def draw_heat_map(transition_matrix, model_name):
    file_path =  "./DTMC_Models/" + model_name + "_heatmap.png"  # Change this to the desired file path
    plt.figure(figsize=(6, 6))
    plt.imshow(transition_matrix, cmap='Blues', interpolation='nearest')

    # Adding labels
    num_states = transition_matrix.shape[0]
    plt.xticks(np.arange(num_states), np.arange(num_states))
    plt.yticks(np.arange(num_states), np.arange(num_states))
    plt.xlabel("To State")
    plt.ylabel("From State")
    plt.title("Transition Matrix Heatmap")

    # Adding color bar for the values
    plt.colorbar()
    plt.savefig(file_path)
    plt.show()


def qoutes(string):
    return "\""+ string + "\""

def indentation():
    return "        "

if __name__ == "__main__":
    generate_model(16,3 ,"RandomModel", 12)
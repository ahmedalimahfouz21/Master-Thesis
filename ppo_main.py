# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 12:11:48 2023

@author: DELL
"""
# import os
import numpy as np
import torch
import torch.optim as optim
from envvironment_creator import env_creator
from batchify import batchify, batchify_obs, unbatchify
from ppo import Agent

gpu_enabled = False
gpu_list = {0:"cuda:0", 1:"cuda:1", 2:"cuda:2", 3:"cuda:3"}
gpu = gpu_list[0]
device = torch.device(gpu if (torch.cuda.is_available() and gpu_enabled) else "cpu")
INPUT_MODELS = {0:'drunkardswalk.xml', 1:'examplerobot.xml', 2:'cleaningrobot.xml', 3:'onlineshopping.xml', 4:"httprequest.xml", 5:'teleassistance.xml', 6:'travelassistance.xml', 7:'RandomModel.xml', 8:"SimpleModel.xml"}
AGENTS_TYPE = {0:"GOOD", 1:"BAD", 2:"OPPONENTS"}
       
if __name__ == "__main__":
    """ALGO PARAMS"""
    ent_coef = 0.1
    vf_coef = 0.1
    clip_coef = 0.1
    gamma = 0.99
    batch_size = 32
    total_episodes = 2
    
    """ ENV SETUP """
    models = [7]
    initial_state = [0]
    end_state = [14]
    model_files = []
    for k in models:
       model_file = './DTMC_Models/' + INPUT_MODELS[k]   
       model_files.append(model_file)
    env,_ = env_creator(model_files, AGENTS_TYPE[2], initial_state, end_state )
    num_agents = len(env.possible_agents)
    num_actions = env.num_actions
    observation_size = 1
    max_cycles = env.game_object.get_max_steps() + 3
    stack_size = env.game_object.get_num_system_objects()
    """ LEARNER SETUP """
    ppo_agent = Agent(env=env, in_dim=observation_size, out_dim=num_actions).to(device)
    optimizer = {}
    for agent in env.possible_agents:
        optimizer[agent] = optim.Adam(ppo_agent.NeuralNetworks[agent].parameters(), lr=0.001, eps=1e-5)
    
    """ ALGO LOGIC: EPISODE STORAGE"""
    end_step = 0
    total_episodic_return = 0
    
    rb_obs = {}
    rb_actions = {}
    rb_logprobs = {}
    rb_rewards = {}
    rb_terms = {}
    rb_values = {}
    rb_invalid_action_masks = {}

    for agent in env.possible_agents:
        rb_obs[agent] = torch.zeros((max_cycles, 1, 1)).to(device)
        rb_actions[agent] = torch.zeros((max_cycles, 1)).to(device) 
        rb_logprobs[agent] = torch.zeros((max_cycles, 1)).to(device)   
        rb_rewards[agent] = torch.zeros((max_cycles, 1)).to(device)   
        rb_terms[agent] = torch.zeros((max_cycles, 1)).to(device)    
        rb_values[agent] = torch.zeros((max_cycles, 1)).to(device)
        rb_invalid_action_masks[agent] = torch.zeros((max_cycles, 1) + (agent.get_num_valid_actions(),)).to(device)

    """ TRAINING LOGIC """
    # train for n number of episodes
    for episode in range(total_episodes):
        # collect an episode
        with torch.no_grad():
            # collect observations and convert to batch of torch tensors
            env.reset(seed=None) 
            # reset the episodic return
            total_episodic_return = {}
            for agent in env.possible_agents:
                total_episodic_return[agent] = 0
            # each episode has num_steps
            for step in range(0, max_cycles):
                # rollover the observation
                observations = {}
                actions = {}
                logprobs= {}
                values  = {}
                rewards = {}
                terminations = {}
                invalid_action_masks = {}
                for agent in env.possible_agents:
                    observations[agent] = []
                    actions[agent] = []
                    logprobs[agent] = []
                    values[agent] = []
                    rewards[agent] = []
                    terminations[agent] = []
                    invalid_action_masks[agent] = [] 
                for agent in env.agents:
                    # print(agent.get_name())
                    observation, reward, termination, truncation, info = env.last()
                    # print(observation)
                    if termination or truncation:
                       action = None
                    else:
                       agent.update_valid_actions_mask(agent.get_system())
                       #print(observation)
                       obs = batchify_obs(observation, agent, device)
                       invalid_action_masks[agent].append(agent.get_valid_actions_mask())
                       action_mask = {agent:agent.get_valid_actions_mask()}
                       action_mask = batchify(action_mask,device)
                       print(obs)
                       action, logprob, _, value = ppo_agent.get_action_and_value(obs, agent, invalid_action_masks = action_mask)
                       action = unbatchify(action)[0]
                       logprob = unbatchify(logprob)
                       value = unbatchify(value)
                       actions[agent].append(action) 
                       logprobs[agent].append(logprob)
                       values[agent].append(value)  
                       observations[agent].append(observation)
                    env.step(action)                        
                if (action != None):
                   for agent in env.agents:
                       rewards[agent].append(env.rewards[agent]) 
                       terminations[agent].append(env.terminations[agent])
                   for agent in env.possible_agents:
                       rb_obs[agent][step] = batchify({agent:observations[agent]}, device) 
                       rb_rewards[agent][step] = batchify({agent:rewards[agent]}, device)
                       rb_terms[agent][step] = batchify({agent:terminations[agent]}, device)
                       rb_actions[agent][step] = batchify({agent:actions[agent]},device)
                       rb_logprobs[agent][step] = batchify({agent:logprobs[agent]},device)
                       rb_values[agent][step] = batchify({agent:values[agent]},device).flatten()
                       rb_invalid_action_masks[agent][step] = batchify({agent:invalid_action_masks[agent]},device)

                   # compute episodic return
                   for agent in env.possible_agents:
                      total_episodic_return[agent] += rb_rewards[agent][step].cpu().numpy()

                if all([env.terminations[a] for a in env.terminations]) or all([env.terminations[a] for a in env.terminations]):
                    end_step = step
                    break

        # bootstrap value if not done
        with torch.no_grad():
            rb_advantages = {}
            rb_returns = {}
            for agent in env.possible_agents:
               rb_advantages[agent] = torch.zeros_like(rb_rewards[agent]).to(device)
               for t in reversed(range(end_step)):
                   delta = (
                       rb_rewards[agent][t]
                       + gamma * rb_values[agent][t + 1] * rb_terms[agent][t + 1]
                       - rb_values[agent][t]
                   )
                   rb_advantages[agent][t] = delta + gamma * gamma * rb_advantages[agent][t + 1]
               rb_returns[agent] = rb_advantages[agent] + rb_values[agent]
        # convert our episodes to batch of individual transitions
        b_obs = {}
        b_logprobs = {}
        b_actions = {}
        b_returns = {}
        b_values = {}
        b_advantages = {}
        b_invalid_action_masks = {}
        for agent in env.possible_agents:
            b_obs[agent] = torch.flatten(rb_obs[agent][:end_step], start_dim=0, end_dim=1)
            b_logprobs[agent] = torch.flatten(rb_logprobs[agent][:end_step], start_dim=0, end_dim=1)
            b_actions[agent] = torch.flatten(rb_actions[agent][:end_step], start_dim=0, end_dim=1)
            b_returns[agent] = torch.flatten(rb_returns[agent][:end_step], start_dim=0, end_dim=1)
            b_values[agent] = torch.flatten(rb_values[agent][:end_step], start_dim=0, end_dim=1)
            b_advantages[agent] = torch.flatten(rb_advantages[agent][:end_step], start_dim=0, end_dim=1)
            b_invalid_action_masks[agent] = torch.flatten(rb_invalid_action_masks[agent][:end_step], start_dim=0, end_dim=1)
       
        # Optimizing the policy and value network
        b_index = {}
        batch_index = {}
        for agent in env.possible_agents:
           b_index[agent] = np.arange(len(b_actions[agent]))
           clip_fracs = []
           for repeat in range(3):
               # shuffle the indices we use to access the data
               np.random.shuffle(b_index[agent])
               for start in range(0, len(b_actions[agent]), batch_size):
                   # select the indices we want to train on
                   end = start + batch_size
                   batch_index[agent] = b_index[agent][start:end]   
                   _, newlogprob, entropy, value = ppo_agent.get_action_and_value(
                       b_obs[agent][batch_index[agent]], agent,
                       action = b_actions[agent].long()[batch_index[agent]], 
                       invalid_action_masks = b_invalid_action_masks[agent][batch_index[agent]]
                   )
                   logratio = newlogprob - b_logprobs[agent][batch_index[agent]]
                   ratio = logratio.exp()
                   
                   with torch.no_grad():
                      # calculate approx_kl http://joschu.net/blog/kl-approx.html
                       old_approx_kl = (-logratio).mean()
                       approx_kl = ((ratio - 1) - logratio).mean()
                       clip_fracs += [
                           ((ratio - 1.0).abs() > clip_coef).float().mean().item()
                       ]

                   # normalize advantaegs
                   advantages = b_advantages[agent][batch_index[agent]]
                   advantages = (advantages - advantages.mean()) / (
                       advantages.std() + 1e-8
                   )

                   # Policy loss
                   pg_loss1 = -b_advantages[agent][batch_index[agent]] * ratio
                   pg_loss2 = -b_advantages[agent][batch_index[agent]] * torch.clamp(
                       ratio, 1 - clip_coef, 1 + clip_coef
                   )
                   pg_loss = torch.max(pg_loss1, pg_loss2).mean()
   
                   # Value loss
                   value = value.flatten()
                   v_loss_unclipped = (value - b_returns[agent][batch_index[agent]]) ** 2
                   v_clipped = b_values[agent][batch_index[agent]] + torch.clamp(
                       value - b_values[agent][batch_index[agent]],
                       -clip_coef,
                       clip_coef,
                   )
                   v_loss_clipped = (v_clipped - b_returns[agent][batch_index[agent]]) ** 2
                   v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                   v_loss = 0.5 * v_loss_max.mean()

                   entropy_loss = entropy.mean()
                   loss = pg_loss - ent_coef * entropy_loss + v_loss * vf_coef

                   optimizer[agent].zero_grad()
                   loss.backward()
                   optimizer[agent].step()

           y_pred, y_true = b_values[agent].cpu().numpy(), b_returns[agent].cpu().numpy()
           var_y = np.var(y_true)
           explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y
                     
           print("**" + agent.get_name() + "**")
           print(f"Training episode {episode}")
           print(f"Episodic Return: {np.mean(total_episodic_return[agent])}")
           print(f"Episode Length: {end_step}")
           print(f"Episode Final State: {env.game_object.get_current_state()}")
           print("")
           print(f"Value Loss: {v_loss.item()}")
           print(f"Policy Loss: {pg_loss.item()}")
           print(f"Old Approx KL: {old_approx_kl.item()}")
           print(f"Approx KL: {approx_kl.item()}")
           print(f"Clip Fraction: {np.mean(clip_fracs)}")
           print(f"Explained Variance: {explained_var.item()}")
           print("\n***************************************\n")
        print("\n-------------------------------------------\n")
    
    PATH = './PPO_Agent'
    # Save our model if it's time
    for agent in env.possible_agents:
        network = ppo_agent.NeuralNetworks[agent]
        torch.save(network.actor.state_dict(), PATH + '/ppo_actor_' + agent.get_name() +'.pth')
        torch.save(network.critic.state_dict(), PATH + '/ppo_critic_' + agent.get_name() +'.pth')
    
    print(f"Device: {device}")
    
    """ RENDER THE POLICY """
    ppo_agent.eval()

    with torch.no_grad():
        # render 5 episodes out
        for episode in range(5):
            env.reset(seed=None)
            observation, reward, termination, truncation, info = env.last()
            obs = batchify_obs(observation, env.agent_selection, device)
            terms = [termination]
            truncs = [truncation]
            i=0
            while not any(terms) and not any(truncs):
              terms = [False]
              truncs = [False]
              for agent in env.agents:
                 agent.update_valid_actions_mask(agent.get_system())
                 action_mask = {agent:agent.get_valid_actions_mask()}
                 action_mask = batchify(action_mask,device)
                 action, logprob, _, value = ppo_agent.get_action_and_value(obs, agent, invalid_action_masks = action_mask)
                 player_action = unbatchify(action)[0]
                 env.step(player_action)
                 observation, reward, termination, truncation, info = env.last()
                 obs = batchify_obs(observation, agent, device)
                 terms.append(termination)
                 truncs.append(truncation)
                 i=i+1
    
            for player in env.agents:
               print(player.get_name())
               print(player.get_score())
            print("\n-------------------------------------------\n") 
    print("Done")
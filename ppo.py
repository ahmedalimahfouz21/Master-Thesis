"""Basic code which shows what it's like to run PPO on the Pistonball env using the parallel API, this code is inspired by CleanRL.

This code is exceedingly basic, with no logging or weights saving.
The intention was for users to have a (relatively clean) ~200 line file to refer to when they want to design their own learning algorithm.

Author: Jet (https://github.com/jjshoots)
"""
# import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
import matplotlib.pyplot as plt
from batchify import batchify, batchify_obs, unbatchify

gpu_enabled = False
gpu_list = {0:"cuda:0", 1:"cuda:1", 2:"cuda:2", 3:"cuda:3"}
gpu = gpu_list[0]
device = torch.device(gpu if (torch.cuda.is_available() and gpu_enabled) else "cpu")

class NeuralNetwork(nn.Module):
    
    def __init__(self, in_dim, out_dim, strategy_type):
        self.strategy_type = strategy_type
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(in_dim, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
        )
        self.actor = nn.Linear(1024, out_dim)
        self.critic = nn.Linear(1024, 1)
        
    def _layer_init(self, layer, std=np.sqrt(2), bias_const=0.0):
        torch.nn.init.orthogonal_(layer.weight, std)
        torch.nn.init.constant_(layer.bias, bias_const)
        return layer
        
# ALGO LOGIC: initialize agent here:
class CategoricalMasked(Categorical):
    def __init__(self, probs=None, logits=None, validate_args=None, masks=[]):
        self.masks = masks
        if len(self.masks) == 0:
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
        else:
            self.masks = masks.type(torch.BoolTensor).to(device)
            logits = torch.where(self.masks, logits, torch.tensor(-1e+8).to(device))
            super(CategoricalMasked, self).__init__(probs, logits, validate_args)
    
    def entropy(self):
        if len(self.masks) == 0:
            return super(CategoricalMasked, self).entropy()
        p_log_p = self.logits * self.probs
        p_log_p = torch.where(self.masks, p_log_p, torch.tensor(0.).to(device))
        return -p_log_p.sum(-1)

class Agent(nn.Module):
    """
        A standard in_dim-64-64-out_dim Feed Forward Neural Network.
    """
    def __init__(self, env, in_dim, out_dim):
        super().__init__()
        self.device = device
        self.env = env
        self.NeuralNetworks = {}
        for agent in env.possible_agents:
            self.NeuralNetworks[agent] = NeuralNetwork(in_dim, out_dim[agent], agent.get_player_strategy_name()).to(self.device)


    def get_value(self,  x, agent, strategy_type):
        hidden = self.NeuralNetworks[strategy_type].network(x / (1.0 * self.env.num_states[agent]))
        return self.NeuralNetworks[strategy_type].critic(hidden)
      

    def get_action_and_value(self, x, agent, action=None, invalid_action_masks=None):
        hidden = self.NeuralNetworks[agent].network(x / (1.0 * self.env.num_states[agent]))
        logits = self.NeuralNetworks[agent].actor(hidden)
        split_logits = torch.split(logits,  1)
        
        if invalid_action_masks is not None:
            split_invalid_action_masks = torch.split(invalid_action_masks, 1)
            multi_categoricals = [CategoricalMasked(logits=logits, masks=iam) for (logits, iam) in zip(split_logits, split_invalid_action_masks)]
        else:
            # print("None")
            multi_categoricals = [Categorical(logits=logits) for logits in split_logits]
       
        if action is None:
            action = torch.stack([categorical.sample() for categorical in multi_categoricals])
        logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)])
        entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals])
        return action, logprob.sum(0), entropy.sum(0), self.NeuralNetworks[agent].critic(hidden)

    def get_random_action(self, invalid_action_masks):
        valid_actions = []
        for i in range(len(invalid_action_masks)):
            if invalid_action_masks[i]:
               valid_actions.append(i)
        action = valid_actions[np.random.randint(0, len(valid_actions))]  
        return action
    
    def learn(self, PATH, env, total_episodes):     
        """ALGO PARAMS"""
        ent_coef = 0.1
        vf_coef = 0.1
        clip_coef = 0.1
        gamma = 0.99
        batch_size = 64
        n_epochs = 3

 
        # Train the actor and critic networks. Here is where the main PPO algorithm resides.
        print("Learning...")
        
        """ ENV SETUP """
        max_cycles = env.game_object.get_max_steps() + 2

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
 
        """ LEARNER SETUP """
        optimizer = {}
        for agent in env.possible_agents:
            optimizer[agent] = optim.Adam(self.NeuralNetworks[agent].parameters(), lr=0.001, eps=1e-5)
        
        """ Algorithm behaviour """
        policy_losses = {}
        value_function_loss = {}
        explained_variance = {}
        entropy_losses = {}
        for agent in env.agents:
            policy_losses[agent] = []
            value_function_loss[agent] = []
            explained_variance[agent] = []
            entropy_losses[agent] = []

        """ TRAINING LOGIC """
        # train for n number of episodes
        for episode in range(total_episodes):
            # collect an episode
            with torch.no_grad():
                # collect observations and convert to batch of torch tensors
                env.reset(seed=None) 
                # reset the episodic return
                total_episodic_return = {}
                for agent in env.agents:
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
                    for agent in env.agents:
                        observations[agent] = []
                        actions[agent] = []
                        logprobs[agent] = []
                        values[agent] = []
                        rewards[agent] = []
                        terminations[agent] = []
                        invalid_action_masks[agent] = []                    
                    for agent in env.agents:
                        observation, reward, termination, truncation, info = env.last()
                        if termination or truncation:
                           action = None
                           observations[agent].append(observation) 
                        else:
                           agent.update_valid_actions_mask(agent.get_system())
                           obs = batchify_obs(observation, agent, device)
                           invalid_action_masks[agent].append(agent.get_valid_actions_mask())  
                           action_mask = {agent:agent.get_valid_actions_mask()}
                           action_mask = batchify(action_mask,device)
                           action, logprob, _, value = self.get_action_and_value(obs, agent, invalid_action_masks = action_mask)
                           action = unbatchify(action)[0]
                           logprob = unbatchify(logprob)
                           value = unbatchify(value)
                           actions[agent].append(action)
                           logprobs[agent].append(logprob)
                           values[agent].append(value)
                           observations[agent].append(observation)
                        env.step(action)                    
                    if (action != None):
                       for agent in env.possible_agents:
                           rewards[agent].append(env.rewards[agent]) 
                           terminations[agent].append(env.terminations[agent] )
                       for agent in env.possible_agents:
                           rb_obs[agent][step] = batchify({agent:observations[agent]}, device)
                           rb_rewards[agent][step] = batchify({agent:rewards[agent]}, device)
                           rb_terms[agent][step] = batchify({agent:terminations[agent]}, device)
                           rb_actions[agent][step] = batchify({agent:actions[agent]},device)
                           rb_logprobs[agent][step] = batchify({agent:logprobs[agent]},device)
                           rb_values[agent][step] = batchify({agent:values[agent]},device).flatten()
                           rb_invalid_action_masks[agent][step] = batchify({agent: invalid_action_masks[agent]},device)
                    # compute episodic return
                    for agent in env.possible_agents:
                       total_episodic_return[agent] += rb_rewards[agent][step].cpu().numpy()
                    # if we reach termination or truncation, end
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
               b_index[agent] = np.arange(len(b_obs[agent]))
               clip_fracs = []
               for repeat in range(n_epochs):
                 # shuffle the indices we use to access the data
                 np.random.shuffle(b_index[agent])
                 for start in range(0, len(b_obs[agent]), batch_size):
                    # select the indices we want to train on
                    end = start + batch_size
                    batch_index[agent] = b_index[agent][start:end]   
                    _, newlogprob, entropy, value = self.get_action_and_value(
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
                    pg_loss1 = advantages * ratio
                    pg_loss2 = advantages * torch.clamp(
                        ratio, 1 - clip_coef, 1 + clip_coef
                    )
                    pg_loss = -torch.min(pg_loss1, pg_loss2).mean()
                      
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
               
               #LOGS
               policy_losses[agent].append(threashold(pg_loss.item(),-10,10))
               value_function_loss[agent].append(threashold(v_loss.item(),-1000,1000))
               explained_variance[agent].append(threashold(explained_var.item(),-10,10))
               entropy_losses[agent].append(threashold(entropy_loss.item(),-100,100))
               
               print("**" + agent.get_name() + "**")
               print(f"Training episode {episode}")
               print(f"Episodic Return: {np.mean(total_episodic_return[agent])}")                     
               print("")
               print(f"Value Loss: {v_loss.item()}")
               print(f"Policy Loss: {pg_loss.item()}")
               print(f"Old Approx KL: {old_approx_kl.item()}")
               print(f"Approx KL: {approx_kl.item()}")
               print(f"Clip Fraction: {np.mean(clip_fracs)}")
               print(f"Explained Variance: {explained_var.item()}")
               print("\n***************************************\n")
            print(f"Episode Length: {end_step}") 
            print(f"Cummlative undesirableness: {env.game_object.get_undesirableness()}")
            print(f"Episode Final State: {env.game_object.get_current_state()}")
            print("\n-------------------------------------------\n") 
            self.eval()        
            if (episode + 1) % 100 == 0:
               # Save our model if it's time
               for agent in env.possible_agents:
                   network = self.NeuralNetworks[agent]
                   torch.save(network.actor.state_dict(), PATH + '/ppo_actor_'+ agent.get_name() +'.pth')
                   torch.save(network.critic.state_dict(), PATH + '/ppo_critic_'+ agent.get_name()  +'.pth')
                   
            if (episode + 1) % 1000 == 0:
               # Plot training curves
               window_size = 5
               for agent in env.possible_agents:
                   
                   plt.figure(figsize=(12, 4))
                   
                   plt.subplot(1, 4, 1)
                   plt.plot(explained_variance[agent])
                   plt.xlabel('Epoch')
                   plt.ylabel('Explained Variance')
                   plt.title('Ex. Variance')
                                    
                   plt.subplot(1, 4, 1)
                   plt.plot(moving_average(explained_variance[agent],window_size=window_size))
                   plt.xlabel('Epoch')
                   plt.ylabel('Explained Variance')
                   plt.title(' Ex. Variance')
                                    
                   plt.subplot(1, 4, 2)
                   plt.plot(policy_losses[agent])
                   plt.xlabel('Epoch')
                   plt.ylabel('Policy Loss')
                   plt.title('Policy Loss')
                                   
                   plt.subplot(1, 4, 2)
                   plt.plot(moving_average(policy_losses[agent],window_size=window_size))
                   plt.xlabel('Epoch')
                   plt.ylabel('Policy Loss')
                   plt.title('Policy Loss')
                                    
                   plt.subplot(1, 4, 3)
                   plt.plot(value_function_loss[agent])
                   plt.xlabel('Epoch')
                   plt.ylabel('Value Loss')
                   plt.title('Value Loss')
                                   
                   plt.subplot(1, 4, 3)
                   plt.plot(moving_average(value_function_loss[agent],window_size=window_size))
                   plt.xlabel('Epoch')
                   plt.ylabel('Value Loss')
                   plt.title('Value Loss')
                   
                   plt.subplot(1, 4, 4)
                   plt.plot(entropy_losses[agent])
                   plt.xlabel('Epoch')
                   plt.ylabel('Entropy Loss')
                   plt.title('Ent. Loss')
                                   
                   plt.subplot(1, 4, 4)
                   plt.plot(moving_average(entropy_losses[agent],window_size=window_size))
                   plt.xlabel('Epoch')
                   plt.ylabel('Entropy Loss')
                   plt.title('Ent. Loss')
                   
                   figure_file = PATH + "/PPO_Analysis/" + agent.get_name() + '.png'
                   plt.savefig(figure_file) 

                   # plt.tight_layout()
                   # plt.show()
            

    def render_policy(self, PATH, env):
         """ RENDER THE POLICY """
         self.eval()

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
                     action, logprob, _, value = self.get_action_and_value(obs, agent, invalid_action_masks = action_mask)
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
                     
def train(PATH, env, actor_model, critic_model, total_episodes):
    print("Training", flush=True)
    model = create_ppo_agent(env, actor_model, critic_model)   
    # Train the PPO model with a specified total timesteps
    print(model.device)
    model.learn(PATH, env, total_episodes = total_episodes)  
    # model.render_policy(PATH,env)
    
def create_ppo_agent(env, actor_model, critic_model):
      num_actions = env.num_actions
      observation_size = 1
      # Create a model for PPO.
      model = Agent(env=env, in_dim=observation_size, out_dim=num_actions).to(device)
      # Tries to load in an existing actor/critic model to continue training on
      if actor_model != '' and critic_model != '':
         print(f"Loading in {actor_model} and {critic_model}...", flush=True) 
         for agent in env.possible_agents:
             network = model.NeuralNetworks[agent]
             network.actor.load_state_dict(torch.load(actor_model + agent.get_name() +'.pth'))
             network.critic.load_state_dict(torch.load(critic_model + agent.get_name() +'.pth'))
         print("Successfully loaded.", flush=True)
      elif actor_model != '' or critic_model != '': # Don't train from scratch if user accidentally forgets actor/critic model
         print("Error: Either specify both actor/critic models or none at all. We don't want to accidentally override anything!")
         return None
      else:
         print("Training from scratch.", flush=True)
      return model

def threashold(val, minval, maxval):
    if  val > maxval:
        return maxval
    elif val < minval:
        return minval
    else: 
        return val


def moving_average(data, window_size):
    smoothed_data = []
    for i in range(len(data)):
        if i < window_size // 2:
            smoothed_value = sum(data[:i + window_size // 2 + 1]) / (i + window_size // 2 + 1)
        elif i >= len(data) - window_size // 2:
            smoothed_value = sum(data[i - window_size // 2:]) / (len(data) - i + window_size // 2)
        else:
            smoothed_value = sum(data[i - window_size // 2:i + window_size // 2 + 1]) / window_size
        smoothed_data.append(smoothed_value)
    return smoothed_data
  


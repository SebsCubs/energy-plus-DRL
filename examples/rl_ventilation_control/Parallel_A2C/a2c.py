import logging
import os
import sys
import traceback
import matplotlib
matplotlib.use('Agg')  # For saving in a headless program. Must be before importing matplotlib.pyplot or pylab!
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
import torch

class A2C_trainer:
    def __init__(self, actor_critic_policy, config):
        self.config = config
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.lr = config['learning_rate']
        self.model = actor_critic_policy
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.states, self.actions, self.rewards = [], [], []
        self.scores, self.episodes, self.average = [], [], []
        self.episode = 0
        self.score = 0
        self.EPISODES = config['number_of_episodes']
        self.max_average = -99999999999
        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path):
            os.makedirs(self.Save_Path)
        self.Model_name = config['model_path']
        self.verbose = config['eplus_verbose']


       
    def discount_rewards(self, rewards):
        gamma = 0.99
        running_add = 0
        discounted_r = np.zeros_like(rewards)
        for i in reversed(range(0, len(rewards))):
            running_add = running_add * gamma + rewards[i]
            discounted_r[i] = running_add
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r) + 1e-10  # Add epsilon to avoid division by zero
        return discounted_r

    def replay(self, experience):
        try:
            # Convert experience data to tensors
            states = torch.FloatTensor(np.vstack(experience['states']))
            actions = torch.LongTensor(np.vstack(experience['actions']))
            self.score = np.sum(experience['rewards'])
            discounted_r = torch.FloatTensor(self.discount_rewards(experience['rewards']))
            
            # Forward pass to get action probabilities and values
            action_probs, values = self.model(states)
            values = values.squeeze()
            
            # Ensure action_probs and actions have compatible shapes
            action_probs = action_probs.gather(1, actions)
            
            # Calculate advantages
            advantages = discounted_r - values
            
            # Calculate actor and critic loss
            actor_loss = -((torch.log(action_probs) * actions).sum(dim=1) * advantages).mean()
            #actor_loss = -(torch.log(action_probs) * advantages).sum(dim=1).mean()
            critic_loss = advantages.pow(2).mean()
            
            # Backpropagation
            loss = actor_loss + critic_loss
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5)  # Gradient clipping
            self.optimizer.step()
            
            # Clear states, actions, and rewards
            self.states, self.actions, self.rewards = [], [], []
            
        except Exception as e:
            error_message = f"An error occurred during the replay: {e}\n{traceback.format_exc()}"
            print(error_message)
            logging.error(error_message)

    
    def save(self, suffix=""):
        try:       
            torch.save(self.model.state_dict(), f"{self.Model_name[:-4]}{suffix}.pth")
        except Exception as e:
            error_message = f"Failed to save model: {e}\n{traceback.format_exc()}"
            print(error_message)
            logging.error(error_message)
    
    def evaluate_model(self):
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        
        if str(self.episode)[-1:] == "0":
            try:      
                fig, ax = plt.subplots()              
                ax.plot(self.episodes, self.scores, 'b')
                ax.plot(self.episodes, self.average, 'r')
                ax.set_ylabel('Score', fontsize=18)
                ax.set_xlabel('Steps', fontsize=18)
                ax.set_title("Episode scores")
                fig.savefig(f"{self.Model_name[:-4]}.png")
                plt.close('all')
            except OSError as e:
                print(e)
            except:
                e = sys.exc_info()[0]
                print("Something else went wrong e: ", e)                                   
        if self.average[-1] >= self.max_average:
            self.max_average = self.average[-1]
            self.save(suffix="_best")
            SAVING = "SAVING"
        else:
            SAVING = ""
        logging.info("episode: {}/{}, score: {}, average: {:.2f}, max average:{:.2f} {}".format(self.episode, self.EPISODES, self.scores[-1], self.average[-1],self.max_average, SAVING))

        return self.average[-1]

    def update(self, experience):
        """
        Update the model using the provided experience.
        :param experience: Dictionary containing 'states', 'actions', 'rewards', and 'episode'
        """
        try:
            self.replay(experience)
            self.episode = experience['episode']
            self.episodes.append(self.episode)
            self.scores.append(self.score)
            self.evaluate_model()
        except Exception as e:
            error_message = f"An error occurred during update: {e}\n{traceback.format_exc()}"
            print(error_message)
            logging.error(error_message)
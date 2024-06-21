import logging
import os
import sys
import traceback
from matplotlib import pyplot as plt
import numpy as np
import torch.optim as optim
import torch
import torch.nn as nn

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
        self.score = 0
        self.episode = 0
        self.EPISODES = config['number_of_episodes']
        self.max_average = -99999999999
        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path):
            os.makedirs(self.Save_Path)
        self.Model_name = config['model_path']
        self.verbose = config['eplus_verbose']


       
    def discount_rewards(self, reward):
        gamma = 0.99
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0, len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
        discounted_r -= np.mean(discounted_r)
        discounted_r /= np.std(discounted_r)
        return discounted_r

    def replay(self, experience):
        try:
            states = torch.FloatTensor(np.vstack(experience.states))
            actions = torch.FloatTensor(np.vstack(experience.actions))
            self.score = np.sum(experience.rewards)
            discounted_r = torch.FloatTensor(self.discount_rewards(experience.rewards))
            action_probs, values = self.model(states)
            values = values.squeeze()
            action_probs = torch.gather(action_probs, 1, actions.long())
            advantages = discounted_r - values
            actor_loss = -(torch.log(action_probs) * actions).sum(dim=1) * advantages
            critic_loss = advantages.pow(2)
            loss = actor_loss.mean() + critic_loss.mean()
            self.optimizer.zero_grad()
            loss.backward()
            if self.verbose == 0:
                pass
            elif self.verbose == 1:
                pass
            elif self.verbose == 2:
                for name, param in self.model.named_parameters():
                    if param.grad is not None:
                        print(f"Gradients for {name}: {param.grad}")
            else:
                print("Warning: Verbose should be 0, 1 or 2 only")

            self.optimizer.step()

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
        print("episode: {}/{}, score: {}, average: {:.2f}, max average:{:.2f} {}".format(self.episode, self.EPISODES, self.scores[-1], self.average[-1],self.max_average, SAVING))

        return self.average[-1]

    ## Entry point for A2C algo ##
    def update(self, experience):
        #Experiences contain: states, actions and rewards -> a big tensor containing all for an episode
        self.replay(experience)
        self.episode += 1
        self.episodes.append(self.episode)
        self.scores.append(self.score)
        self.evaluate_model()
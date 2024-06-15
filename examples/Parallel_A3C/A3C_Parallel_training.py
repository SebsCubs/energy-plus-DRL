"""
Author: Sebastian Cubides
"""
import configparser
from multiprocessing import Pool, Manager
from multiprocessing.managers import BaseManager
from pathlib import Path
import shutil
import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from eplus_drl import EmsPy, BcaEnv
import datetime
import time
import matplotlib
matplotlib.use('Agg') # For saving in a headless program. Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

start_time = time.time()
# -- FILE PATHS --
script_directory = os.path.dirname(os.path.abspath(__file__))

def load_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    config_dict = {
        'ep_path': config['DEFAULT']['ep_path'],
        'idf_file_name': config['DEFAULT']['idf_file_name'],
        'ep_weather_path': config['DEFAULT']['ep_weather_path'],
        'cvs_output_path': config['DEFAULT']['cvs_output_path'],
        'number_of_subprocesses': config.getint('DEFAULT', 'number_of_subprocesses'),
        'number_of_episodes': config.getint('DEFAULT', 'number_of_episodes'),
        'eplus_verbose': config.getint('DEFAULT', 'eplus_verbose'),
        'state_size': tuple(map(int, config['DEFAULT']['state_size'].split(','))),
        'action_size': config.getint('DEFAULT', 'action_size'),
        'learning_rate': config.getfloat('DEFAULT', 'learning_rate'),
        'model_path': config['DEFAULT']['model_path']
    }
    
    return config_dict

####################### RL model and class  #################
class ActorCriticModel(nn.Module):
    def __init__(self, input_shape, action_space):
        super(ActorCriticModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(np.prod(input_shape), 512)
        self.fc2 = nn.Linear(512, action_space)
        self.fc3 = nn.Linear(512, 1)
        
        self.softmax = nn.Softmax(dim=-1)
        self.elu = nn.ELU()
        
    def forward(self, x):
        x = self.flatten(x)
        x = self.elu(self.fc1(x))
        action_probs = self.softmax(self.fc2(x))
        state_value = self.fc3(x)
        return action_probs, state_value

def create_A2C_model(input_shape, action_space, lr):
    model = ActorCriticModel(input_shape, action_space)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    return model, optimizer

class A2C_agent:
    def __init__(self, config):
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.lr = config['learning_rate']
        self.model, self.optimizer = create_A2C_model(self.state_size, self.action_size, self.lr)
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
        self.state = None
        self.time_of_day = None
    
    def copy(self):
        new = A2C_agent(config)
        new.model.load_state_dict(self.model.state_dict())
        new.optimizer = optim.Adam(new.model.parameters(), lr=self.lr)
        new.episode = self.episode
        return new

    def get_EPISODES(self):
        return self.EPISODES
    
    def update_global(self, global_a2c_agent):
        global_a2c_agent.update_global_net(self.model)
        global_a2c_agent.append_score(self.score)
        global_a2c_agent.evaluate_model()

    def update_global_net(self, model):
        self.model.load_state_dict(model.state_dict())
    
    def append_score(self, score):
        self.episode += 1
        self.episodes.append(self.episode)
        self.scores.append(score)
    
    def remember(self, state, action, reward):
        self.states.append(state)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)
    
    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.model(state)
        action = np.random.choice(self.action_size, p=action_probs.numpy().squeeze())
        return action
        
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

    def replay(self):
        states = torch.FloatTensor(np.vstack(self.states))
        actions = torch.FloatTensor(np.vstack(self.actions))
        discounted_r = torch.FloatTensor(self.discount_rewards(self.rewards))
        action_probs, values = self.model(states)
        values = values.squeeze()
        advantages = discounted_r - values

        actor_loss = -(torch.log(action_probs) * actions).sum(dim=1) * advantages
        critic_loss = advantages.pow(2)

        loss = actor_loss.mean() + critic_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        self.states, self.actions, self.rewards = [], [], []

    def save(self):
        torch.save(self.model.state_dict(), self.Model_name)
    
    def evaluate_model(self):
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if __name__ == "__main__":
            if str(self.episode)[-1:] == "0":
                try:      
                    fig, ax = plt.subplots()              
                    ax.plot(self.episodes, self.scores, 'b')
                    ax.plot(self.episodes, self.average, 'r')
                    ax.set_ylabel('Score', fontsize=18)
                    ax.set_xlabel('Steps', fontsize=18)
                    ax.set_title("Episode scores")
                    fig.savefig(os.path.join(self.Save_Path, self.path)+".png")
                    plt.close('all')
                except OSError as e:
                    print(e)
                except:
                    e = sys.exc_info()[0]
                    print("Something else went wrong e: ", e)                                   
            if self.average[-1] >= self.max_average:
                self.max_average = self.average[-1]
                self.save()
                SAVING = "SAVING"
            else:
                SAVING = ""
            print("episode: {}/{}, score: {}, average: {:.2f}, max average:{:.2f} {}".format(self.episode, self.EPISODES, self.scores[-1], self.average[-1],self.max_average, SAVING))

        return self.average[-1]
####################### Process function: EnergyPlus manager #################

class Energyplus_manager:
    def __init__(self, episode, a2c_object, config, lock):
        self.global_a2c_object = a2c_object
        self.local_a2c_object = self.global_a2c_object.copy()
        self.config = config
        self.lock = lock

        self.episode = episode
        self.a2c_state = None
        self.step_reward = 0  
        self.episode_reward = 0
        self.previous_state = None
        self.previous_action = None
        self.zn0 = 'Thermal Zone 1' #name of the zone to control 
        self.tc_intvars = {}  # empty, don't need any
        self.tc_vars = {
            'zn0_temp': ('Zone Air Temperature', self.zn0),  # deg C
            'air_loop_fan_mass_flow_var' : ('Fan Air Mass Flow Rate','FANSYSTEMMODEL VAV'),  # kg/s
            'air_loop_fan_electric_power' : ('Fan Electricity Rate','FANSYSTEMMODEL VAV'),  # W
            'deck_temp_setpoint' : ('System Node Setpoint Temperature','Node 30'),  # deg C
            'deck_temp' : ('System Node Temperature','Node 30'),  # deg C
            'ppd' : ('Zone Thermal Comfort Fanger Model PPD', 'THERMAL ZONE 1 189.1-2009 - OFFICE - WHOLEBUILDING - MD OFFICE - CZ4-8 PEOPLE'),
        }
        self.tc_meters = {} # empty, don't need any
        self.tc_weather = {
            'oa_rh': ('outdoor_relative_humidity'),  # %RH
            'oa_db': ('outdoor_dry_bulb'),  # deg C
            'oa_pa': ('outdoor_barometric_pressure'),  # Pa
            'sun_up': ('sun_is_up'),  # T/F
            'rain': ('is_raining'),  # T/F
            'snow': ('is_snowing'),  # T/F
            'wind_dir': ('wind_direction'),  # deg
            'wind_speed': ('wind_speed')  # m/s
        }
        # ACTION SPACE
        self.tc_actuators = {
            'fan_mass_flow_act': ('Fan', 'Fan Air Mass Flow Rate', 'FANSYSTEMMODEL VAV'),  # kg/s
        }
        self.calling_point_for_callback_fxn = EmsPy.available_calling_points[7]  
        self.sim_timesteps = 6  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)
        self.working_dir = BcaEnv.get_temp_run_dir()
        #--- Copy Energyplus into a temp folder ---#
        self.directory_name = "Energyplus_temp"
        self.eplus_copy_path = os.path.join(self.working_dir, self.directory_name)
        self.delete_directory(self.directory_name)
        shutil.copytree(self.config['ep_path'], self.eplus_copy_path)
        self.sim = BcaEnv(
            ep_path=self.eplus_copy_path,
            ep_idf_to_run=self.config['idf_file_name'],
            timesteps=self.sim_timesteps,
            tc_vars=self.tc_vars,
            tc_intvars=self.tc_intvars,
            tc_meters=self.tc_meters,
            tc_actuator=self.tc_actuators,
            tc_weather=self.tc_weather
        )
        self.sim.set_calling_point_and_callback_function(
            calling_point=self.calling_point_for_callback_fxn,
            observation_function=self.observation_function,  # optional function
            actuation_function=self.actuation_function,  # optional function
            update_state=True,  # use this callback to update the EMS state
            update_observation_frequency=1,  # linked to observation update
            update_actuation_frequency=1  # linked to actuation update
        )

    def run_episode(self):
        if self.config['eplus_verbose'] == 2:
            self.run_simulation()
        elif self.config['eplus_verbose'] == 1:
            self.run_simulation()    
        elif self.config['eplus_verbose'] == 0 :
            devnull = open(os.devnull, 'w')
            orig_stdout_fd = os.dup(1)
            orig_stderr_fd = os.dup(2)
            os.dup2(devnull.fileno(), 1)
            os.dup2(devnull.fileno(), 2)             
            self.run_simulation()
            os.dup2(orig_stdout_fd, 1) #Restoring stdout
            os.dup2(orig_stderr_fd, 2)
            os.close(orig_stdout_fd)
            os.close(orig_stderr_fd)
        else:
            raise ValueError("eplus_verbose must be 0, 1 or 2")
            
        self.run_neural_net()
        with self.lock:
            self.local_a2c_object.update_global(self.global_a2c_object)
        self.delete_directory()

    def run_neural_net(self):
        self.local_a2c_object.replay() # train the network

    def observation_function(self):
        self.time = self.sim.get_ems_data(['t_datetimes'])
        if self.time < datetime.datetime.now():
            var_data = self.sim.get_ems_data(list(self.sim.tc_var.keys()))
            weather_data = self.sim.get_ems_data(list(self.sim.tc_weather.keys()), return_dict=True)
            self.zn0_temp = var_data[0]  
            self.fan_mass_flow = var_data[1]  # kg/s
            self.fan_electric_power = var_data[2]  # W
            self.deck_temp_setpoint = var_data[3]  # deg C
            self.deck_temp = var_data[4]  # deg C
            self.ppd = var_data[5]  # percent
            self.a2c_state = self.get_state(var_data, weather_data)
            self.step_reward = self.reward_function()
            if(self.previous_state is None):
                self.previous_state = self.a2c_state
                self.previous_action = 0
            self.previous_state = self.a2c_state
            self.local_a2c_object.remember(self.a2c_state, self.previous_action, self.step_reward)
            self.episode_reward += self.step_reward

        return self.step_reward              

    def actuation_function(self): 
        if self.time < datetime.datetime.now():    
            action = self.local_a2c_object.act(self.a2c_state)
            fan_flow_rate = action * (2.18 / 10)
            self.previous_action = action            
        return { 'fan_mass_flow_act': fan_flow_rate }
      
    def run_simulation(self): 
        out_dir = os.path.join(self.working_dir, 'out') 
        self.sim.run_env(self.config['ep_weather_path'], out_dir)
    
    def reward_function(self):
        nomalized_setpoint = (21 - 18) / 17
        alpha = 1
        beta = 1
        reward = - (alpha * abs(nomalized_setpoint - self.a2c_state[1]) + beta * self.a2c_state[3])
        return reward

    def get_state(self, var_data, weather_data):   
        self.time_of_day = self.sim.get_ems_data(['t_hours'])
        weather_data = list(weather_data.values())[:2]
        state = np.concatenate((np.array([self.time_of_day]), var_data, weather_data)) 
        state[0] = state[0] / 24
        state[1] = (state[1] - 18) / 17
        state[2] = state[2] / 2.18
        state[3] = state[3] / 3045.81
        state[4] = (state[4] - 15) / 15
        state[5] = state[5] / 35
        state[6] = state[6] / 100
        state[7] = state[7] / 100
        state[8] = (state[8] + 10) / 20
        return state
        
    def delete_directory(self, temp_folder_name=""):
        directory_path = os.path.join(self.working_dir, temp_folder_name)
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        out_path = Path('out')
        if out_path.exists() and out_path.is_dir():
            shutil.rmtree(out_path)


def run_eplus_train_thread(episode, config, lock, shared_a2c_object):
    eplus_object = Energyplus_manager(episode, shared_a2c_object, config, lock)
    eplus_object.run_episode()
    return eplus_object.episode_reward


if __name__ == "__main__":
    pid = os.getpid()
    print("Main process, pid: ", pid)
    config = load_config()
    CustomManager = BaseManager
    CustomManager.register('A2C_agent', A2C_agent)   
    with Manager() as global_manager:
        lock = global_manager.Lock()
        with CustomManager() as manager:
            shared_a2c_object = manager.A2C_agent(config)
            EPISODES = shared_a2c_object.get_EPISODES()
            with Pool(processes=config['number_of_subprocesses'], maxtasksperchild=3) as pool:
                for index in range(EPISODES):
                    pool.apply_async(run_eplus_train_thread, args=(index, config, lock, shared_a2c_object))
                pool.close()
                pool.join()

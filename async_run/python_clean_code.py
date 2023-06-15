"""
Author: Sebastian Cubides
"""
from multiprocessing import Pool, Manager
from multiprocessing.managers import BaseManager
from pathlib import Path
import shutil
import os
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)
os.environ["KMP_AFFINITY"] = "noverbose"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
tf.autograph.set_verbosity(3)
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)
import sys
import numpy as np
from emspy import EmsPy, BcaEnv
import datetime
import time
from keras.models import Model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam
import matplotlib
matplotlib.use('Agg') # For saving in a headless program. Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
start_time = time.time()
# -- FILE PATHS --
ep_path = '/usr/local/EnergyPlus-22-1-0'
script_directory = os.path.dirname(os.path.abspath(__file__))
idf_file_name = r'xxxxxxxxx' 
ep_weather_path = r'xxxxxxxx'
cvs_output_path = r'xxxxxxxxx'
####################### RL model and class  #################
def A2CModels(input_shape, action_space, lr):
    X_input = Input(shape=input_shape)
    X = Flatten(input_shape=input_shape)(X_input)
    X_hid2 = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)
    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X_hid2)
    value = Dense(1, kernel_initializer='he_uniform')(X_hid2)
    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss='categorical_crossentropy', optimizer=Adam(learning_rate=lr))
    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=Adam(learning_rate=lr))
    return Actor, Critic

class A2C_agent:
    def __init__(self):
        self.state_size = (9,1)
        self.action_size = 10
        self.lr = 0.001
        self.Actor, self.Critic = A2CModels(input_shape = self.state_size, action_space = self.action_size, lr=self.lr)
        self.ale_easter_egg = 9042023
        self.states, self.actions, self.rewards = [], [], []
        self.scores, self.episodes, self.average = [], [], []
        self.score = 0
        self.episode = 0
        self.EPISODES = 1000
        self.max_average = -99999999999
        self.Save_Path = 'Models'
        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A3C_{}'.format(datetime.datetime.now().strftime("%Y%m%d-%H%M%S"), self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)
        self.state = None
        self.time_of_day = None

    def copy(self):
        new = A2C_agent()
        new.Actor.set_weights(self.Actor.get_weights())
        new.Critic.set_weights(self.Critic.get_weights())
        new.episode = self.episode
        return new

    def get_EPISODES(self):
        return self.EPISODES
    
    def update_global(self, global_a2c_agent):
        #DO NOT USE with the global object, only process copies
        global_a2c_agent.update_global_net(self.Actor,self.Critic)
        global_a2c_agent.append_score(self.score)
        global_a2c_agent.evaluate_model()

    def update_global_net(self,Actor,Critic):
        self.Actor.set_weights(Actor.get_weights())
        self.Critic.set_weights(Critic.get_weights())
    def append_score(self,score):
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
        # Use the network to predict the next action to take, using the model
        state = np.expand_dims(state, axis=0) #Need to add a dimension to the state to make it a 2D array                 
        prediction = self.Actor(state)
        action = np.random.choice(self.action_size, p=np.squeeze(prediction))#Squeeze to remove the extra dimension
        return action
        
    def discount_rewards(self, reward):
        gamma = 0.99    # discount rate
        running_add = 0
        discounted_r = np.zeros_like(reward)
        for i in reversed(range(0,len(reward))):
            running_add = running_add * gamma + reward[i]
            discounted_r[i] = running_add
        discounted_r -= np.mean(discounted_r) # normalizing the result
        discounted_r /= np.std(discounted_r) # divide by standard deviation
        return discounted_r

    def replay(self):
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        self.score = np.sum(self.rewards)
        discounted_r = self.discount_rewards(self.rewards)
        values = self.Critic(states)[:, 0]
        advantages = discounted_r - values
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
    
    def evaluate_model(self):
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))
        if __name__ == "__main__":
            if str(self.episode)[-1:] == "0":# much faster than episode % 100
                try:      
                    fig, ax = plt.subplots() # fig : figure object, ax : Axes object              
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
            # saving best models
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
    def __init__(self, episode, a2c_object:A2C_agent,lock):
        self.global_a2c_object = a2c_object
        self.local_a2c_object = self.global_a2c_object.copy()
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
        # 6-16 valid for timestep loop during simulation, check documentation as this is VERY unpredictable
        self.sim_timesteps = 6  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)
        self.working_dir = BcaEnv.get_temp_run_dir()
        #--- Copy Energyplus into a temp folder ---#
        self.directory_name = "Energyplus_temp"
        self.eplus_copy_path = os.path.join(self.working_dir, self.directory_name)
        self.delete_directory(self.directory_name)
        shutil.copytree(ep_path, self.eplus_copy_path)
        self.sim = BcaEnv(
            ep_path=self.eplus_copy_path,
            ep_idf_to_run=idf_file_name,
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
            actuation_function= self.actuation_function,  # optional function
            update_state=True,  # use this callback to update the EMS state
            update_observation_frequency=1,  # linked to observation update
            update_actuation_frequency=1  # linked to actuation update
        )
        devnull = open('/dev/null', 'w')#To make e+ shut up!
        orig_stdout_fd = os.dup(1)
        orig_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        self.run_simulation()
        os.dup2(orig_stdout_fd, 1) #Restoring stdout
        os.dup2(orig_stderr_fd, 2)
        os.close(orig_stdout_fd)
        os.close(orig_stderr_fd)
        self.run_neural_net()
        with lock:
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
            self.a2c_state = self.get_state(var_data,weather_data)
            self.step_reward = self.reward_function()
            if(self.previous_state is None):
                self.previous_state = self.a2c_state
                self.previous_action = 0
            self.previous_state = self.a2c_state
            self.local_a2c_object.remember(self.a2c_state,self.previous_action,self.step_reward)
            self.episode_reward += self.step_reward

        return self.step_reward              

    def actuation_function(self):     
        action = self.local_a2c_object.act(self.a2c_state)
        fan_flow_rate = action*(2.18/10)
        self.previous_action = action            
        return { 'fan_mass_flow_act': fan_flow_rate, }
      
    def run_simulation(self): 
        out_dir = os.path.join(self.working_dir, 'out') 
        self.sim.run_env(ep_weather_path, out_dir)
    def reward_function(self):
        nomalized_setpoint = (21-18)/17
        alpha = 1
        beta = 1
        #reward = - (  np.square(alpha *(abs(self.a2c_state[6]-0.1))) + beta*(self.a2c_state[3])  ) #based on comfort metric
        reward = - (  alpha*(abs(nomalized_setpoint-self.a2c_state[1])) + beta*self.a2c_state[3]) #Temperature
        return reward

    def get_state(self,var_data, weather_data):   
        #State:                  MAX:                  MIN:
        # 0: time of day        24                    0
        # 1: zone0_temp         35                    18
        # 2: fan_mass_flow      2.18                  0
        # 3: fan_electric_power 3045.81               0
        # 4: deck_temp_setpoint 30                    15
        # 5: deck_temp          35                    0
        # 6: ppd                100                   0        
        # 7: outdoor_rh         100                   0  
        # 8: outdoor_temp       10                    -10
        self.time_of_day = self.sim.get_ems_data(['t_hours'])
        weather_data = list(weather_data.values())[:2]
        state = np.concatenate((np.array([self.time_of_day]),var_data,weather_data)) 
        state[0] = state[0]/24
        state[1] = (state[1]-18)/17
        state[2] = state[2]/2.18
        state[3] = state[3]/3045.81
        state[4] = (state[4]-15)/15
        state[5] = state[5]/35
        state[6] = state[6]/100
        state[7] = state[7]/100
        state[8] = (state[8]+10)/20
        return state
        
    def delete_directory(self,temp_folder_name = ""):
        directory_path = os.path.join(self.working_dir, temp_folder_name)
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        out_path = Path('out')
        if out_path.exists() and out_path.is_dir():
            shutil.rmtree(out_path)

class CustomManager(BaseManager):
    pass
def run_one_manger(ep,a2c_object,lock):
    eplus_object = Energyplus_manager(ep,a2c_object,lock)
    return eplus_object.episode_reward

if __name__ == "__main__":
    pid = os. getpid()
    print("Main process, pid: ",pid)
    start_time = time.time()
    CustomManager.register('A2C_agent', A2C_agent)   
    with Manager() as global_manager:
        lock = global_manager.Lock()
        with CustomManager() as manager:
            shared_a2c_object = manager.A2C_agent()
            EPISODES = shared_a2c_object.get_EPISODES()
            with Pool(processes=10, maxtasksperchild = 3) as pool:
                for index in range(EPISODES):
                    pool.apply_async(run_one_manger, args=(index, shared_a2c_object, lock,))
                pool.close()
                pool.join()
            end_time = time.time()
            print("Time to run ",EPISODES," episodes: ", end_time - start_time)
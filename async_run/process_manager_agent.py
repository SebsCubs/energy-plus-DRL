"""
Author: Sebastian Cubides

"""
from multiprocessing import Pool, Manager
from multiprocessing.managers import BaseManager
from pathlib import Path
import shutil
import os

#Supress TF warnings:
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
from keras.optimizers import Adam, RMSprop

import matplotlib
matplotlib.use('Agg') # For saving in a headless program. Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt

import tkinter
import copy


start_time = time.time()

# -- FILE PATHS --
# * E+ Download Path *
ep_path = '/usr/local/EnergyPlus-22-1-0'  # path to E+ on system
script_directory = os.path.dirname(os.path.abspath(__file__))

# IDF File / Modification Paths
idf_file_name = r'/home/jun/HVAC/repo_recover/energy-plus-DRL/BEMFiles/sdu_damper_all_rooms.idf'  # building energy model (BEM) IDF file
# Weather Path
ep_weather_path = r'/home/jun/HVAC/energy-plus-DRL/BEMFiles/DNK_Jan_Feb.epw'  # EPW weather file
# Output .csv Path (optional)
cvs_output_path = r'/home/jun/HVAC/energy-plus-DRL/Dataframes/dataframes_output_train.csv'


####################### RL model and class  #################
#It holds: 
# The global network
# THe episode count
# Saves the best models
# The scores of every episode
# Any metrics on the network's performance
# All interactions with the object must be done with methods, OOP standard

def A2CModels(input_shape, action_space, lr):
    X_input = Input(shape=input_shape)
    X = Flatten(input_shape=input_shape)(X_input)

    #X_hid = Dense(64, activation="elu", kernel_initializer='he_uniform')(X)
    X_hid2 = Dense(512, activation="elu", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X_hid2)
    value = Dense(1, kernel_initializer='he_uniform')(X_hid2)

    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss='categorical_crossentropy')

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse')

    return Actor, Critic

class A2C_agent:
    def __init__(self):
        # -- RL AGENT --
        #Input: TC variables + outdoor humidity and temperature + time of day (3) 
        #Output: 10 possible actions for the fan mass flow rate
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
        self.max_average = -99999999999 #To save the best model

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

    def get_episode(self):
        return self.episode

    def get_episodes_array(self):
        return self.episodes
    def get_scores(self):
        return self.scores
    
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
        # store episode actions to memory
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
        # Compute the gamma-discounted rewards over an episode
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
        # reshape memory to appropriate shape for training
        states = np.vstack(self.states)
        actions = np.vstack(self.actions)
        # Compute discounted rewards
        self.score = np.sum(self.rewards)
        discounted_r = self.discount_rewards(self.rewards)
        # Get Critic network predictions
        values = self.Critic(states)[:, 0]
        # Compute advantages
        advantages = discounted_r - values
        # training Actor and Critic networks
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
        #reset training memory
        self.states, self.actions, self.rewards = [], [], []


    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        #self.Critic.save(self.Model_name + '_Critic.h5')
    

    def evaluate_model(self):

        #self.episodes.append(self.episode)
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
    """
    Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
    scope throughout the simulation.
    Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
    (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
    instance.
    """
    def __init__(self, episode, a2c_object:A2C_agent,lock):
        
        #--- A2C Reinforcement learning Agent ---#
        self.global_a2c_object = a2c_object
        self.local_a2c_object = self.global_a2c_object.copy()
        self.episode = episode
        self.a2c_state = None
        self.step_reward = 0  
        self.episode_reward = 0
        self.previous_state = None
        self.previous_action = None

        #--- STATE SPACE (& Auxiliary Simulation Data)
        self.zn0 = 'Thermal Zone 1' #name of the zone to control 
        self.tc_intvars = {}  # empty, don't need any

        self.tc_vars = {
            # Building
            #'hvac_operation_sched': ('Schedule Value', 'HtgSetp 1'),  # is building 'open'/'close'?
            # -- Zone 0 (Core_Zn) --
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
            # HVAC Control Setpoints
            'fan_mass_flow_act': ('Fan', 'Fan Air Mass Flow Rate', 'FANSYSTEMMODEL VAV'),  # kg/s
        }

        # -- Simulation Params --
        self.calling_point_for_callback_fxn = EmsPy.available_calling_points[7]  # 6-16 valid for timestep loop during simulation, check documentation as this is VERY unpredictable
        self.sim_timesteps = 6  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)


        # simulation data state
        self.zn0_temp = None  # deg C
        self.fan_mass_flow = None  # kg/s
        self.fan_electric_power = None  # W
        self.deck_temp_setpoint = None  # deg C
        self.deck_temp = None  # deg C
        self.ppd = None  # percent
        self.time = None

        self.working_dir = BcaEnv.get_temp_run_dir()
        #print(f"Thread: Running at working dir: {self.working_dir}")
        
        #--- Copy Energyplus into a temp folder ---#

        self.directory_name = "Energyplus_temp"
        # Define the path of the target directory
        self.eplus_copy_path = os.path.join(self.working_dir, self.directory_name)
        #Delete directory if it exists
        self.delete_directory(self.directory_name)
        # Copy the directory to the target directory
        shutil.copytree(ep_path, self.eplus_copy_path)

        # -- Create Building Energy Simulation Instance --
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
        
        #To make e+ shut up!
        
        devnull = open('/dev/null', 'w')
        orig_stdout_fd = os.dup(1)
        orig_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
    
        self.run_simulation()
        
        #Restoring stdout
        os.dup2(orig_stdout_fd, 1)
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
        # -- FETCH/UPDATE SIMULATION DATA --
        self.time = self.sim.get_ems_data(['t_datetimes'])
        #check that self.time is less than current time
        if self.time < datetime.datetime.now():
            # Get data from simulation at current timestep (and calling point) using ToC names
            var_data = self.sim.get_ems_data(list(self.sim.tc_var.keys()))
            weather_data = self.sim.get_ems_data(list(self.sim.tc_weather.keys()), return_dict=True)

            # get specific values from MdpManager based on name
            self.zn0_temp = var_data[0]  
            self.fan_mass_flow = var_data[1]  # kg/s
            self.fan_electric_power = var_data[2]  # W
            self.deck_temp_setpoint = var_data[3]  # deg C
            self.deck_temp = var_data[4]  # deg C
            self.ppd = var_data[5]  # percent
            # OR if using "return_dict=True"
            #outdoor_temp = weather_data['oa_db']  # outdoor air dry bulb temp


            # -- UPDATE STATE & REWARD ---
            self.a2c_state = self.get_state(var_data,weather_data)
            self.step_reward = self.reward_function()
                
            # Initialize previous state for first step
            if(self.previous_state is None):
                self.previous_state = self.a2c_state
                self.previous_action = 0

            self.previous_state = self.a2c_state
            self.local_a2c_object.remember(self.a2c_state,self.previous_action,self.step_reward)
            self.episode_reward += self.step_reward

        return self.step_reward              

    def actuation_function(self):        
        #RL control
        #The action is a list of values for each actuator
        #The fan flow rate actuator is the only one for now
        #It divides the range of operation into 10 discrete values, with the first one being 0
        # In energyplus, the max flow rate is depending in the mass flow rate and a density depending of 
        # the altitude and 20 deg C -> a safe bet is dnsity = 1.204 kg/m3
        # The max flow rate of the fan is autosized to 1.81 m3/s
        # The mass flow rate is in kg/s, so the max flow rate is 1.81*1.204 = 2.18 kg/s        
        #Agent action
        action = self.local_a2c_object.act(self.a2c_state)         
        #Map the action to the fan mass flow rate
        fan_flow_rate = action*(2.18/10)
        self.previous_action = action
        """
        # print reporting       
        current_time = self.sim.get_ems_data(['t_cumulative_time']) 
        print(f'\n\nHours passed: {str(current_time)}')
        print(f'Time: {str(self.time)}')
        print('\n\t* Actuation Function:')
        print(f'\t\Actor action: {action}')
        print(f'\t\Fan mass flow rate: {self.fan_mass_flow}'  # outputs ordered list
            f'\n\t\Action:{fan_flow_rate}')  # outputs dictionary
        print(f'\t\Last State: {self.state}')        
        print(f'\t\Reward: {self.step_reward}')
        """
            
        return { 'fan_mass_flow_act': fan_flow_rate, }
    
    def run_simulation(self): 
        out_dir = os.path.join(self.working_dir, 'out') 
        self.sim.run_env(ep_weather_path, out_dir)

    def reward_function(self):
        #Taking into account the fan power and the deck temp, a no-occupancy scenario
        #State:                  MAX:                  MIN:
        # 1: zone0_temp         35                    18
        # 3: fan_electric_power 3045.81               0
        # 6: ppd                100                   0
        # State is already normalized
        nomalized_setpoint = (21-18)/17
        alpha = 0.8
        beta = 1.2
        #reward = - (  np.square(alpha *(abs(self.a2c_state[6]-0.1))) + beta*(self.a2c_state[3])  ) #occupancy, based on comfort metric
        reward = - (  alpha*(abs(nomalized_setpoint-self.a2c_state[1])) + beta*self.a2c_state[3]) #No ocupancy
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
        
        #concatenate self.time_of_day , var_data and weather_data
        state = np.concatenate((np.array([self.time_of_day]),var_data,weather_data)) 

        #normalize each value in the state according to the table above
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
        # Define the path of the directory to be deleted
        directory_path = os.path.join(self.working_dir, temp_folder_name)
        # Delete the directory if it exists
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            #print(f"Directory '{directory_path}' deleted")    

        out_path = Path('out')
        if out_path.exists() and out_path.is_dir():
            shutil.rmtree(out_path)  
            #print(f"Directory '{out_path}' deleted")


####################### Custom manager #################
#This allows to share a global object holding the networks of the RL Agent.

class CustomManager(BaseManager):
    pass




def run_one_manger(ep,a2c_object,lock):
    eplus_object = Energyplus_manager(ep,a2c_object,lock)
    return eplus_object.episode_reward

if __name__ == "__main__":
    pid = os. getpid()
    print("Main process, pid: ",pid)
    start_time = time.time()
    # register the a python class with the custom manager
    CustomManager.register('A2C_agent', A2C_agent)
    
    with Manager() as global_manager:
        # create the shared lock instance
        lock = global_manager.Lock()

        # create and start the custom manager
        with CustomManager() as manager:
            # create a shared python object
            shared_a2c_object = manager.A2C_agent()

            EPISODES = shared_a2c_object.get_EPISODES()
            # start worker processes
            with Pool(processes=10, maxtasksperchild = 3) as pool:
                #--- Process handler ###

                for index in range(EPISODES):
                    pool.apply_async(run_one_manger, args=(index, shared_a2c_object, lock,))
                
                pool.close()
                pool.join()
            
            end_time = time.time()
            print("Time to run ",EPISODES," episodes: ", end_time - start_time)
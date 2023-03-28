"""
Author: Sebastian Cubides

"""
import sys

sys.path.insert(0, 'C:\EnergyPlusV22-1-0')
import os
from pyenergyplus import api #Importing from folder, a warning may show
from pyenergyplus.api import EnergyPlusAPI
import numpy as np
from emspy import EmsPy, BcaEnv
import pylab
import datetime
import time
import matplotlib.pyplot as plt
from keras.models import Model, load_model
from keras.layers import Input, Dense, Flatten
from keras.optimizers import Adam, RMSprop
from keras import backend as K


start_time = time.time()

# -- FILE PATHS --
# * E+ Download Path *
ep_path = 'C:\EnergyPlusV22-1-0'  # path to E+ on system
# IDF File / Modification Paths
idf_file_name = r'C:\Projects\SDU\Thesis\pyenergyplus\BEMFiles\sdu_double_heating.idf'  # building energy model (BEM) IDF file
# Weather Path
ep_weather_path = r'C:\Projects\SDU\Thesis\pyenergyplus\BEMFiles\DNK_Jan_Feb.epw'  # EPW weather file
# Output .csv Path (optional)
cvs_output_path = r'C:\Projects\SDU\Thesis\pyenergyplus\Dataframes\dataframes_output_train.csv'






def A2CModels(input_shape, action_space, lr):

    X_input = Input(shape=input_shape)
    X = Flatten(input_shape=input_shape)(X_input)

    X_hid = Dense(10, activation="elu", kernel_initializer='he_uniform')(X)

    action = Dense(action_space, activation="softmax", kernel_initializer='he_uniform')(X_hid)
    value = Dense(1, kernel_initializer='he_uniform')(X_hid)

    Actor = Model(inputs = X_input, outputs = action)
    Actor.compile(loss='categorical_crossentropy', optimizer=RMSprop(learning_rate=lr))

    Critic = Model(inputs = X_input, outputs = value)
    Critic.compile(loss='mse', optimizer=RMSprop(learning_rate=lr))

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

        # Instantiate games and plot memory
        self.states, self.actions, self.rewards = [], [], []
        self.scores, self.episodes, self.average = [], [], []
        self.EPISODES = 100
        self.max_average = -99999999999 #To save the best model
        self.Save_Path = 'Models'

        if not os.path.exists(self.Save_Path): os.makedirs(self.Save_Path)
        self.path = '{}_A2C_{}'.format("SDU_Building", self.lr)
        self.Model_name = os.path.join(self.Save_Path, self.path)
        self.state = None
    
        self.time_of_day = None

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
        discounted_r = self.discount_rewards(self.rewards)
        # Get Critic network predictions
        values = self.Critic.predict(states)[:, 0]
        # Compute advantages
        advantages = discounted_r - values
        # training Actor and Critic networks
        self.Actor.fit(states, actions, sample_weight=advantages, epochs=1, verbose=0)
        self.Critic.fit(states, discounted_r, epochs=1, verbose=0)
        # reset training memory
        self.states, self.actions, self.rewards = [], [], []

    def load(self, Actor_name, Critic_name):
        self.Actor = load_model(Actor_name, compile=False)
        #self.Critic = load_model(Critic_name, compile=False)

    def save(self):
        self.Actor.save(self.Model_name + '_Actor.h5')
        #self.Critic.save(self.Model_name + '_Critic.h5')
    

    def evaluate_model(self, score, episode):
        self.scores.append(score)
        self.episodes.append(episode)
        self.average.append(sum(self.scores[-50:]) / len(self.scores[-50:]))


        #if str(episode)[-1:] == "0":# much faster than episode % 100
        pylab.plot(self.episodes, self.scores, 'b')
        pylab.plot(self.episodes, self.average, 'r')
        pylab.ylabel('Score', fontsize=18)
        pylab.xlabel('Steps', fontsize=18)
        try:
            pylab.savefig(self.path+".png")
        except OSError:
            pass
        
        # saving best models
        if self.average[-1] >= self.max_average:
            self.max_average = self.average[-1]
            self.save()
            SAVING = "SAVING"
        else:
            SAVING = ""
        print("episode: {}/{}, score: {}, average: {:.2f} {}".format(episode, self.EPISODES, score, self.average[-1], SAVING))

        return self.average[-1]

    

class Energyplus_manager:
    """
    Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
    scope throughout the simulation.
    Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
    (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
    instance.
    """
    def __init__(self, agent:A2C_agent):

        self.a2c_agent = agent #Any change to the self.a2c_agent object will change the original instance
        self.a2c_state = None
        self.step_reward = 0  
        self.previous_state = None
        self.previous_action = None

            # STATE SPACE (& Auxiliary Simulation Data)
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

        # -- Create Building Energy Simulation Instance --
        self.sim = BcaEnv(
            ep_path=ep_path,
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

    def observation_function(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        self.time = self.sim.get_ems_data(['t_datetimes'])
        #check that self.time is less than current time
        if self.time < datetime.datetime.now():
            # Get data from simulation at current timestep (and calling point) using ToC names
            var_data = self.sim.get_ems_data(list(self.sim.tc_var.keys()))
            weather_data = self.sim.get_ems_data(list(self.sim.tc_weather.keys()), return_dict=True)

            callbacks.append(1)

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
            
            self.a2c_agent.remember(self.a2c_state, self.previous_action, self.step_reward)

            self.previous_state = self.a2c_state

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
        action = a2c_agent.act(self.a2c_state)         
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
    
    def reward_function(self):
        #Taking into account the fan power and the deck temp, a no-occupancy scenario
        #State:                  MAX:                  MIN:
        # 1: zone0_temp         35                    18
        # 3: fan_electric_power 3045.81               0
        # State is already normalize, hyperparameters are all equal to one for now 
        nomalized_setpoint = (21-18)/17
        reward = - (abs(nomalized_setpoint-self.a2c_state[1]) + self.a2c_state[3])
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
        
   


if __name__ == "__main__":
    #  --- Create agent instance --
    callbacks = []
    episode_rewards = []
    a2c_agent = A2C_agent()

    end_time = time.time()
    print("Time to initialize: ", end_time - start_time)

    for ep in range(a2c_agent.EPISODES):
        start_time = time.time()
        eplus_manager = Energyplus_manager(a2c_agent)  
        eplus_manager.sim.run_env(ep_weather_path)    
        episode_rewards.append(np.sum(a2c_agent.rewards))

        print("Episode:", (ep+1), "Reward:", episode_rewards[-1])
        end_time = time.time()
        print("Time to run episode: ", end_time - start_time)

        start_time = time.time()
        average = a2c_agent.evaluate_model(episode_rewards[-1], (ep+1)) # evaluate the model
        print("No. of callbacks: ", len(callbacks), "Empsy callbacks: ", eplus_manager.sim.callback_current_count)
        a2c_agent.replay() # train the network      
        callbacks = []
        # -- Sample Output Data --
        output_dfs = eplus_manager.sim.get_df(to_csv_file=cvs_output_path)  # LOOK at all the data collected here, custom DFs can be made too, possibility for creating a CSV file (GB in size)
        del eplus_manager #delete object
        

        end_time = time.time()
        print("Time to evaluate and train: ", end_time - start_time)

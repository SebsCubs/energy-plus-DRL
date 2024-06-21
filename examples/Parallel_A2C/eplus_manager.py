import logging
from pathlib import Path
import shutil
import os
import numpy as np
from eplus_drl import EmsPy, BcaEnv
import datetime
import matplotlib
matplotlib.use('Agg')  # For saving in a headless program. Must be before importing matplotlib.pyplot or pylab!

"""
Main function of the manager: Run an EnergyPlus simulation, controlled by the RL agent's Policy
To do this, 4 main sections interact together: 
1. EmsPy (eplus wrapper RL oriented)
2. Control Policy (A torch model with an implementation of an act() function).
3. State and action space dictionaries (Policy's Input/Output) with its embeddings.
4. RL-agent's reward function
"""

class Energyplus_manager:
    #EmsPy dictionaries (Will change for every eplus control strategy)
    tc_vars = {
            'zn0_temp': ('Zone Air Temperature', 'Thermal Zone 1'),
            'air_loop_fan_mass_flow_var' : ('Fan Air Mass Flow Rate','FANSYSTEMMODEL VAV'),
            'air_loop_fan_electric_power' : ('Fan Electricity Rate','FANSYSTEMMODEL VAV'),
            'deck_temp_setpoint' : ('System Node Setpoint Temperature','Node 30'),
            'deck_temp' : ('System Node Temperature','Node 30'),
            'ppd' : ('Zone Thermal Comfort Fanger Model PPD', 'THERMAL ZONE 1 189.1-2009 - OFFICE - WHOLEBUILDING - MD OFFICE - CZ4-8 PEOPLE'),
        }
    tc_weather = {
            'oa_rh': ('outdoor_relative_humidity'),
            'oa_db': ('outdoor_dry_bulb'),
            'oa_pa': ('outdoor_barometric_pressure'),
            'sun_up': ('sun_is_up'),
            'rain': ('is_raining'),
            'snow': ('is_snowing'),
            'wind_dir': ('wind_direction'),
            'wind_speed': ('wind_speed')
        }
    tc_actuators = {
            'fan_mass_flow_act': ('Fan', 'Fan Air Mass Flow Rate', 'FANSYSTEMMODEL VAV'),
        }
    #Not used in this example:
    tc_intvars = {}
    tc_meters = {}
    ## 

    def __init__(self, episode, control_policy, config):
        self.local_policy = control_policy
        self.config = config
        self.episode = episode
        self.a2c_state = None
        self.step_reward = 0  
        self.previous_state = None
        self.previous_action = None
        self.action_size = config['action_size']
        self.states, self.actions, self.rewards = [], [], []
        
        self.setup_logging()
        self.setup_emspy_environment()
        

    def setup_logging(self):
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s %(levelname)s: %(message)s',
            handlers=[
                logging.FileHandler("a2c_example_program.log"),
                logging.StreamHandler()
            ]
        )

    def setup_emspy_environment(self):
        self.calling_point_for_callback_fxn = EmsPy.available_calling_points[7]
        self.sim_timesteps = 6
        self.working_dir = BcaEnv.get_temp_run_dir()
        self.directory_name = "Energyplus_temp"
        self.eplus_copy_path = os.path.join(self.working_dir, self.directory_name)
        self.delete_directory(self.directory_name)
        shutil.copytree(self.config['ep_path'], self.eplus_copy_path)
        self.calling_point_for_callback_fxn = EmsPy.available_calling_points[7]
        self.sim_timesteps = 6
        self.working_dir = BcaEnv.get_temp_run_dir()
        self.directory_name = "Energyplus_temp"
        self.eplus_copy_path = os.path.join(self.working_dir, self.directory_name)
        self.delete_directory(self.directory_name)
        shutil.copytree(self.config['ep_path'], self.eplus_copy_path)

        #Simulation object (sim):
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
            observation_function=self.observation_function,
            actuation_function=self.actuation_function,
            update_state=True,
            update_observation_frequency=1,
            update_actuation_frequency=1
        )

    def delete_directory(self, temp_folder_name=""):
        directory_path = os.path.join(self.working_dir, temp_folder_name)
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
        out_path = Path('out')
        if out_path.exists() and out_path.is_dir():
            shutil.rmtree(out_path)

    def remember(self, state, action, reward):
        self.states.append(state)
        action_onehot = np.zeros([self.action_size])
        action_onehot[action] = 1
        self.actions.append(action_onehot)
        self.rewards.append(reward)

    def reward_function(self):
        nomalized_setpoint = (21 - 18) / 17
        alpha = 1
        beta = 1
        reward = - (alpha * abs(nomalized_setpoint - self.a2c_state[1]) + beta * self.a2c_state[3])
        return reward

    def normalize_state(self, state):
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

    def get_state(self, var_data, weather_data):   
        self.time_of_day = self.sim.get_ems_data(['t_hours'])
        weather_data = list(weather_data.values())[:2]
        state = np.concatenate((np.array([self.time_of_day]), var_data, weather_data)) 
        return self.normalize_state(state)
    
    def observation_function(self):
        self.time = self.sim.get_ems_data(['t_datetimes'])
        #To skip warming up and pre-simulation routines from energyplus
        if self.time < datetime.datetime.now(): 
            var_data = self.sim.get_ems_data(list(self.tc_vars.keys()))
            weather_data = self.sim.get_ems_data(list(self.tc_weather.keys()), return_dict=True)
            self.a2c_state = self.get_state(var_data, weather_data)
            self.step_reward = self.reward_function()
            if self.previous_state is None:
                self.previous_state = self.a2c_state
                self.previous_action = 0
            self.remember(self.a2c_state, self.previous_action, self.step_reward)
            self.previous_state = self.a2c_state
        return self.step_reward           

    def actuation_function(self): 
        if self.time < datetime.datetime.now():    
            action = self.local_policy.act(self.a2c_state)
            fan_flow_rate = action * (2.18 / 10)
            self.previous_action = action            
        return { 'fan_mass_flow_act': fan_flow_rate }
       
    def run_simulation(self): 
        out_dir = os.path.join(self.working_dir, 'out') 
        self.sim.run_env(self.config['ep_weather_path'], out_dir)
    
    def silence_simulation(self):
        devnull = open(os.devnull, 'w')
        orig_stdout_fd = os.dup(1)
        orig_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)
        try:
            self.run_simulation()
        finally:
            os.dup2(orig_stdout_fd, 1)
            os.dup2(orig_stderr_fd, 2)
            os.close(orig_stdout_fd)
            os.close(orig_stderr_fd)

    
    def run_episode(self):
        """Entry point for running a single episode."""
        if self.config['eplus_verbose'] == 2:
            self.run_simulation()
        elif self.config['eplus_verbose'] == 1:
            self.run_simulation()    
        elif self.config['eplus_verbose'] == 0:
            self.silence_simulation()
        else:
            raise ValueError("eplus_verbose must be 0, 1, or 2")        
        self.delete_directory()


    
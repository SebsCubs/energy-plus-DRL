from policy import Policy
from pathlib import Path
import shutil
import os
import torch
import numpy as np
from eplus_drl import EmsPy, BcaEnv
import datetime
import matplotlib
matplotlib.use('Agg')  # For saving in a headless program. Must be before importing matplotlib.pyplot or pylab!


class Energyplus_manager:
    def __init__(self, episode, control_policy, config):
        self.global_a2c_object = a2c_object
        self.local_a2c_object = self.load_local_model(config)
        self.config = config
        

        self.episode = episode
        self.a2c_state = None
        self.step_reward = 0  
        self.episode_reward = 0
        self.previous_state = None
        self.previous_action = None
        self.zn0 = 'Thermal Zone 1'
        self.tc_intvars = {}
        self.tc_vars = {
            'zn0_temp': ('Zone Air Temperature', self.zn0),
            'air_loop_fan_mass_flow_var' : ('Fan Air Mass Flow Rate','FANSYSTEMMODEL VAV'),
            'air_loop_fan_electric_power' : ('Fan Electricity Rate','FANSYSTEMMODEL VAV'),
            'deck_temp_setpoint' : ('System Node Setpoint Temperature','Node 30'),
            'deck_temp' : ('System Node Temperature','Node 30'),
            'ppd' : ('Zone Thermal Comfort Fanger Model PPD', 'THERMAL ZONE 1 189.1-2009 - OFFICE - WHOLEBUILDING - MD OFFICE - CZ4-8 PEOPLE'),
        }
        self.tc_meters = {}
        self.tc_weather = {
            'oa_rh': ('outdoor_relative_humidity'),
            'oa_db': ('outdoor_dry_bulb'),
            'oa_pa': ('outdoor_barometric_pressure'),
            'sun_up': ('sun_is_up'),
            'rain': ('is_raining'),
            'snow': ('is_snowing'),
            'wind_dir': ('wind_direction'),
            'wind_speed': ('wind_speed')
        }
        self.tc_actuators = {
            'fan_mass_flow_act': ('Fan', 'Fan Air Mass Flow Rate', 'FANSYSTEMMODEL VAV'),
        }
        
        self.calling_point_for_callback_fxn = EmsPy.available_calling_points[7]  
        self.sim_timesteps = 6
        self.working_dir = BcaEnv.get_temp_run_dir()
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
            observation_function=self.observation_function,
            actuation_function=self.actuation_function,
            update_state=True,
            update_observation_frequency=1,
            update_actuation_frequency=1
        )

    def load_local_model(self, config):
        local_model = Policy(config['state_size'], config['action_size'])
        if os.path.exists(local_model.Model_name):
            local_model.model.load_state_dict(torch.load(local_model.Model_name))
        return local_model

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
            os.dup2(orig_stdout_fd, 1)
            os.dup2(orig_stderr_fd, 2)
            os.close(orig_stdout_fd)
            os.close(orig_stderr_fd)
        else:
            raise ValueError("eplus_verbose must be 0, 1 or 2")
            
        self.run_neural_net()
        
        self.local_a2c_object.update_global(self.global_a2c_object)            
        self.delete_directory()

    def run_neural_net(self):
        self.local_a2c_object.replay()

    def observation_function(self):
        self.time = self.sim.get_ems_data(['t_datetimes'])
        if self.time < datetime.datetime.now():
            var_data = self.sim.get_ems_data(list(self.sim.tc_var.keys()))
            weather_data = self.sim.get_ems_data(list(self.sim.tc_weather.keys()), return_dict=True)
            self.zn0_temp = var_data[0]
            self.fan_mass_flow = var_data[1]
            self.fan_electric_power = var_data[2]
            self.deck_temp_setpoint = var_data[3]
            self.deck_temp = var_data[4]
            self.ppd = var_data[5]
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

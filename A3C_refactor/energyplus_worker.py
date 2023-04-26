"""
Author: Sebastian Cubides

"""
import torch.multiprocessing as mp
from pathlib import Path
import shutil
import os
import numpy as np
from emspy import EmsPy, BcaEnv
import datetime
import matplotlib
matplotlib.use('Agg') # For saving in a headless program. Must be before importing matplotlib.pyplot or pylab!
from .discrete_A3C import Net
import tkinter
from utils import v_wrap, set_init, push_and_pull, record

class Energyplus_worker(mp.Process):
    """
    Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
    scope throughout the simulation.
    Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
    (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
    instance.
    """
    def __init__(self, gnet, opt, global_ep, global_ep_r, res_queue, name):
        super(Energyplus_worker, self).__init__()
        self.name = 'w%02i' % name
        self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
        self.gnet, self.opt = gnet, opt
        self.lnet = Net()           # local network, default sizes

        #--- STATE SPACE (& Auxiliary Simulation Data)
        self.zn0 = 'Thermal Zone 1' #name of the zone to control 
        # -- FILE PATHS --
        # * E+ Download Path *
        self.ep_path = '/usr/local/EnergyPlus-22-1-0'  # path to E+ on system
        self.script_directory = os.path.dirname(os.path.abspath(__file__))
        # IDF File / Modification Paths
        self.idf_file_name = r'/home/jun/HVAC/energy-plus-DRL/BEMFiles/sdu_damper_all_rooms.idf'  # building energy model (BEM) IDF file
        # Weather Path
        self.ep_weather_path = r'/home/jun/HVAC/energy-plus-DRL/BEMFiles/DNK_Jan_Feb.epw'  # EPW weather file



        #### RL-EmsPy #####
        # -- Simulation Params --
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
            #'facility_hvac_electricity' : ('Facility Total HVAC Electricity Demand Rate','WHOLE BUILDING'),
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

        self.tc_actuators = {
            # HVAC Control Setpoints
            'fan_mass_flow_act': ('Fan', 'Fan Air Mass Flow Rate', 'FANSYSTEMMODEL VAV'),  # kg/s
        }

        self.calling_point_for_callback_fxn = EmsPy.available_calling_points[7]  # 6-16 valid for timestep loop during simulation, check documentation as this is VERY unpredictable
        self.sim_timesteps = 6  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)
        #--- Copy Energyplus into a temp folder ---#
        self.working_dir = BcaEnv.get_temp_run_dir() #Creates a temporal directory in /tmp
        self.directory_name = "Energyplus_temp"
        self.eplus_copy_path = os.path.join(self.working_dir, self.directory_name)
        self.delete_directory(self.directory_name)
        shutil.copytree(self.ep_path, self.eplus_copy_path)
        # -- Create Building Energy Simulation Instance --
        self.sim = BcaEnv(
            ep_path=self.eplus_copy_path,
            ep_idf_to_run=self.idf_file_name,
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
 

    def run(self):
        """
        1. Run E+ simulation
        2. Calculate gradients and update global net

        """
        # -- To make e+ shut up! --
        devnull = open('/dev/null', 'w')
        orig_stdout_fd = os.dup(1)
        orig_stderr_fd = os.dup(2)
        os.dup2(devnull.fileno(), 1)
        os.dup2(devnull.fileno(), 2)

        self.run_simulation()
        # -- Restoring stdout -- 
        os.dup2(orig_stdout_fd, 1)
        os.dup2(orig_stderr_fd, 2)
        os.close(orig_stdout_fd)
        os.close(orig_stderr_fd)   

        push_and_pull(self.opt, self.lnet, self.gnet, done, s_, buffer_s, buffer_a, buffer_r, GAMMA)

        self.delete_directory()



    def observation_function(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        self.time = self.sim.get_ems_data(['t_datetimes'])
        #check that self.time is less than current time
        if self.time < datetime.datetime.now():
            # Get data from simulation at current timestep (and calling point) using ToC names
            var_data = self.sim.get_ems_data(list(self.sim.tc_var.keys()))
            weather_data = self.sim.get_ems_data(list(self.sim.tc_weather.keys()), return_dict=True)

            # -- UPDATE STATE & REWARD ---
            self.a2c_state = self.get_state(var_data,weather_data) #also uses data from local a2c object
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
            
        return { 'fan_mass_flow_act': fan_flow_rate, }
    
    def run_simulation(self): 
        out_dir = os.path.join(self.working_dir, 'out') 
        self.sim.run_env(self.ep_weather_path, out_dir)

    def reward_function(self):
        #Taking into account the fan power and the deck temp, a no-occupancy scenario
        #State:                  MAX:                  MIN:
        # 1: zone0_temp         35                    18
        # 3: fan_electric_power 3045.81               0
        # 6: ppd                100                   0
        # State is already normalized, all previous states are saved in the local a2c_object
        #nomalized_setpoint = (21-15)/20
        alpha = 1
        beta = 0.5
        #kappa = 1
        reward = - (  alpha *np.square(( np.maximum(0,(self.a2c_state[6]-0.1))) + beta*(self.a2c_state[3]) )) #Comfort
        #reward = - (  alpha*(max(0, nomalized_setpoint-self.a2c_state[1] )) + beta*self.a2c_state[3] ) #Temperature
        return reward

    def get_state(self, var_data, weather_data):   

        #State:                  MAX:                  MIN:
        # 0: time of day        24                    0
        # 1: zone0_temp         35                    15
        # 2: fan_mass_flow      2.18                  0
        # 3: fan_electric_power 3045.81               0
        # 4: deck_temp_setpoint 30                    15
        # 5: deck_temp          35                    0
        # 6: ppd                100                   0     
        # 7: outdoor_rh         100                   0  
        # 8: outdoor_temp       10                    -10
        # 9: wind direction     360                   0
        # 10: wind speed        20                    0

        self.time_of_day = self.sim.get_ems_data(['t_hours'])
        weather_data1 = list(weather_data.values())[:2]
        weather_data2 = list(weather_data.values())[-2:]
        weather_data = np.concatenate((weather_data1, weather_data2))
   
        state = np.concatenate((np.array([self.time_of_day]),var_data[:6],weather_data)) 

        #normalize each value in the state according to the table above
        state[0] = state[0]/24
        state[1] = (state[1]-15)/20
        state[2] = state[2]/2.18
        state[3] = state[3]/3045.81
        state[4] = (state[4]-15)/15
        state[5] = state[5]/35
        state[6] = state[6]/100
        state[7] = state[7]/100
        state[8] = (state[8]+10)/20
        state[9] = state[9]/360
        state[10] = state[10]/20

        self.local_a2c_object.not_averaged_state.append(state)
        
        # takes last state_window samples and averages it 
        
        w = self.local_a2c_object.state_window

        last_w_states = self.local_a2c_object.not_averaged_state[-w:]

        state_average = sum(last_w_states) / len(last_w_states)

        state[1] = state_average[1]
        state[3] = state_average[3]
        state[6] = state_average[6]
        
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

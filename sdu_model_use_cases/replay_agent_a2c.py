"""
Author: Sebastian Cubides

"""
from multiprocessing import Process, Queue
from a2c_agent import A2C_agent
from pathlib import Path
import shutil
import os
import numpy as np
from emspy import EmsPy, BcaEnv
import datetime
import time



start_time = time.time()

# -- FILE PATHS --
# * E+ Download Path *
ep_path = '/usr/local/EnergyPlus-22-1-0'  # path to E+ on system
script_directory = os.path.dirname(os.path.abspath(__file__))

# IDF File / Modification Paths
idf_file_name = r'/home/jun/HVAC/energy-plus-DRL/BEMFiles/sdu_double_heating.idf'  # building energy model (BEM) IDF file
# Weather Path
ep_weather_path = r'/home/jun/HVAC/energy-plus-DRL/BEMFiles/DNK_Jan_Feb.epw'  # EPW weather file
# Output .csv Path (optional)
cvs_output_path = r'/home/jun/HVAC/energy-plus-DRL/Dataframes/dataframes_output_train.csv'





class Energyplus_manager:
    """
    Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
    scope throughout the simulation.
    Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
    (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
    instance.
    """
    def __init__(self,q:Queue):

        #--- A2C Reinforcement learning Agent ---#
        self.q = q
        self.a2c_state = None
        self.step_reward = 0  
        self.previous_state = None
        self.previous_action = None

        #--- Copy Energyplus into a local folder ---#

        self.directory_name = "Energyplus_temp"
        # Define the path of the target directory
        self.eplus_copy_path = os.path.join(script_directory, self.directory_name)
        #Delete directory if it exists
        self.delete_directory(self.directory_name)
        # Copy the directory to the target directory
        shutil.copytree(ep_path, self.eplus_copy_path)

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

        self.run_simulation()
        
        

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
            
            self.q.put([self.a2c_state, self.previous_action, self.step_reward], block = False)
            #print("Queue size",queue.qsize())
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
    
    def run_simulation(self): 
        self.sim.run_env(ep_weather_path)

    def reward_function(self):
        #Taking into account the fan power and the deck temp, a no-occupancy scenario
        #State:                  MAX:                  MIN:
        # 1: zone0_temp         35                    18
        # 3: fan_electric_power 3045.81               0
        # 6: ppd                100                   0
        # State is already normalized
        #nomalized_setpoint = (21-18)/17
        alpha = 0.8
        beta = 1.2
        reward = - (  np.square(alpha *(abs(self.a2c_state[6]-0.1))) + beta*(self.a2c_state[3])  ) #occupancy, based on comfort metric
        #reward = - (  alpha*(abs(nomalized_setpoint-self.a2c_state[1])) + beta*self.a2c_state[3]) #No ocupancy
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
        
    def delete_directory(self,temp_folder_name):
        # Define the path of the directory to be deleted
        directory_path = os.path.join(script_directory, temp_folder_name)
        # Delete the directory if it exists
        if os.path.exists(directory_path):
            shutil.rmtree(directory_path)
            print(f"Directory '{temp_folder_name}' deleted")
        else:
            print(f"Directory '{temp_folder_name}' does not exist")      

        out_path = Path('out')
        if out_path.exists() and out_path.is_dir():
            shutil.rmtree(out_path)  
            print(f"Directory '{out_path}' deleted")
        else:
            print(f"Directory '{out_path}' does not exist")

if __name__ == "__main__":
    #  --- Create agent instance --
    episode_rewards = []
    a2c_agent = A2C_agent()
    failed_eps = 0
    end_time = time.time()
    print("Time to initialize: ", end_time - start_time)
    
    queue = Queue() #Global queue object for the threads

    for ep in range(a2c_agent.EPISODES):
        start_time = time.time()
        #--- Process handler ###
        proc = Process(target=Energyplus_manager,args=(queue,)) 
        proc.start()
        while(proc.exitcode == None):
            a2c_agent.remember(queue)
        proc.join()
        end_time = time.time()
        print("Time to run episode: ", end_time - start_time) 
        episode_rewards.append(np.sum(a2c_agent.rewards))
        print("Episode:", (ep+1), "Reward:", episode_rewards[-1])
        start_time = time.time()
        average = a2c_agent.evaluate_model(episode_rewards[-1], (ep+1)) # evaluate the model
        a2c_agent.replay() # train the network      
        
        end_time = time.time()
        print("Time to evaluate and train: ", end_time - start_time)
        print("Episodes: ", ep)
    
    print("Failed episodes: ",failed_eps)

            

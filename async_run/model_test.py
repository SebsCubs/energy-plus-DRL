"""
Author: Sebastian Cubides

"""
import time

start_time = time.time()

import numpy as np
from emspy import EmsPy, BcaEnv
import datetime
import matplotlib.pyplot as plt
from keras.models import load_model
import data_frame_analysis as dfa

# -- FILE PATHS --
# * E+ Download Path *
ep_path = '/usr/local/EnergyPlus-22-1-0'  # path to E+ on system
# IDF File / Modification Paths
idf_file_name = r'/home/jun/HVAC/repo_recover/energy-plus-DRL/BEMFiles/sdu_double_heating_dec_test.idf' 
# Weather Path
ep_weather_path = r'/home/jun/HVAC/repo_recover/energy-plus-DRL/BEMFiles/DNK_Dec.epw'  # EPW weather file
# Output .csv Path (optional)
cvs_output_path = r'/home/jun/HVAC/repo_recover/energy-plus-DRL/Dataframes/dataframes_output_model.csv'

model_path = r'/home/jun/HVAC/repo_recover/energy-plus-DRL/async_run/Models/20230520-101314_A3C_0.001_Actor.h5'

#model_path = r'/home/jun/HVAC/energy-plus-DRL/sdu_model_use_cases/Models/SDU_Building_A2C_First_1000eps.h5'

# STATE SPACE (& Auxiliary Simulation Data)

zn0 = 'Thermal Zone 1' #name of the zone to control 
tc_intvars = {}  # empty, don't need any

tc_vars = {
    # Building
    #'hvac_operation_sched': ('Schedule Value', 'HtgSetp 1'),  # is building 'open'/'close'?
    # -- Zone 0 (Core_Zn) --
    'zn0_temp': ('Zone Air Temperature', zn0),  # deg C
    'air_loop_fan_mass_flow_var' : ('Fan Air Mass Flow Rate','FANSYSTEMMODEL VAV'),  # kg/s
    'air_loop_fan_electric_power' : ('Fan Electricity Rate','FANSYSTEMMODEL VAV'),  # W
    'deck_temp_setpoint' : ('System Node Setpoint Temperature','Node 30'),  # deg C
    'deck_temp' : ('System Node Temperature','Node 30'),  # deg C
    'ppd' : ('Zone Thermal Comfort Fanger Model PPD', 'THERMAL ZONE 1 189.1-2009 - OFFICE - WHOLEBUILDING - MD OFFICE - CZ4-8 PEOPLE'),
    'damper_node_flow_rate' : ('System Node Mass Flow Rate','CHANGEOVER BYPASS HW RHT DAMPER OUTLET NODE'),
    'total_hvac_energy' : ('Facility Total HVAC Electricity Demand Rate','WHOLE BUILDING'),
    'damper_coil_heating_rate' : ('Heating Coil Heating Rate','Changeover Bypass HW Rht Coil'),  # W
    'pre_heating_coil_htgrate' : ('Heating Coil Heating Rate','HW Htg Coil'),
}

tc_meters = {} # empty, don't need any

tc_weather = {
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
tc_actuators = {
    # HVAC Control Setpoints
    'fan_mass_flow_act': ('Fan', 'Fan Air Mass Flow Rate', 'FANSYSTEMMODEL VAV'),  # kg/s
}


# -- Simulation Params --
calling_point_for_callback_fxn = EmsPy.available_calling_points[7]  # 6-16 valid for timestep loop during simulation
sim_timesteps = 6  # every 60 / sim_timestep minutes (e.g 10 minutes per timestep)

# -- Create Building Energy Simulation Instance --
sim = BcaEnv(
    ep_path=ep_path,
    ep_idf_to_run=idf_file_name,
    timesteps=sim_timesteps,
    tc_vars=tc_vars,
    tc_intvars=tc_intvars,
    tc_meters=tc_meters,
    tc_actuator=tc_actuators,
    tc_weather=tc_weather
)


class Energyplus_Agent:
    """
    Create agent instance, which is used to create actuation() and observation() functions (both optional) and maintain
    scope throughout the simulation.
    Since EnergyPlus' Python EMS using callback functions at calling points, it is helpful to use a object instance
    (Agent) and use its methods for the callbacks. * That way data from the simulation can be stored with the Agent
    instance.
    """
    def __init__(self, bca: BcaEnv):
        self.bca = bca
        # simulation data state
        self.zn0_temp = None  # deg C
        self.fan_mass_flow = None  # kg/s
        self.fan_electric_power = None  # W
        self.deck_temp_setpoint = None  # deg C
        self.deck_temp = None  # deg C
        self.ppd = None  # percent
        self.time = None

        # -- RL AGENT --
        #Input: TC variables + outdoor humidity and temperature + time of day (3) 
        #Output: 10 possible actions for the fan mass flow rate
        self.state_size = (11,1)
        self.action_size = 10

        self.Actor = self.load(model_path) 
        self.state_window = 3 #How many states in the past will we watch to calculate averages
        self.states = []
        self.not_averaged_state = []
        self.state = None
        self.time_of_day = None

    def observation_function(self):
        # -- FETCH/UPDATE SIMULATION DATA --
        self.time = self.bca.get_ems_data(['t_datetimes'])
        

        #check that self.time is less than current time
        if self.time < datetime.datetime.now():
            # Get data from simulation at current timestep (and calling point) using ToC names
            var_data = self.bca.get_ems_data(list(self.bca.tc_var.keys()))
            weather_data = self.bca.get_ems_data(list(self.bca.tc_weather.keys()), return_dict=True)


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
            self.state = self.get_state(var_data,weather_data)
            self.states.append(self.state)
                  

    def actuation_function(self):        
        #RL control
        #The action is a list of values for each actuator
        #The fan flow rate actuator is the only one for now
        #It divides the range of operation into 10 discrete values, with the first one being 0
        # In energyplus, the max flow rate is depending in the mass flow rate and a density depending of 
        # the altitude and 20 deg C -> a safe bet is dnsity = 1.204 kg/m3
        # The max flow rate of the fan is autosized to 1.81 m3/s
        # The mass flow rate is in kg/s, so the max flow rate is 1.81*1.204 = 2.18 kg/s
        
        if self.time < datetime.datetime.now():
            #Agent action
            action = self.act(self.state)    
         
            #Map the action to the fan mass flow rate
            fan_flow_rate = action*(2.18/10)

        """
        # print reporting       
        current_time = self.bca.get_ems_data(['t_cumulative_time']) 
        print(f'\n\nHours passed: {str(current_time)}')
        print(f'Time: {str(self.time)}')
        print('\n\t* Actuation Function:')
        print(f'\t\Actor action: {action}')
        print(f'\t\Fan mass flow rate: {self.fan_mass_flow}'  # outputs ordered list
            f'\n\t\Action:{fan_flow_rate}')  # outputs dictionary
        print(f'\t\Last State: {self.state}') 
        """

        return { 'fan_mass_flow_act': fan_flow_rate, }

    def get_state(self,var_data, weather_data):   

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

        self.time_of_day = self.bca.get_ems_data(['t_hours'])
   
        state = np.concatenate((np.array([self.time_of_day]),var_data[:6], list(weather_data.values())[:2])) 

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
        return state

    def act(self, state):
        # Use the network to predict the next action to take, using the model
        state = np.expand_dims(state, axis=0) #Need to add a dimension to the state to make it a 2D array                 
        prediction = self.Actor(state)[0]
        action = np.random.choice(self.action_size, p=np.squeeze(prediction))#Squeeze to remove the extra dimension
        return action
        
    def load(self, Actor_name):
        return load_model(Actor_name, compile=False)

#  --- Create agent instance ---
my_agent = Energyplus_Agent(sim)

# --- Set your callback function (observation and/or actuation) function for a given calling point ---
sim.set_calling_point_and_callback_function(
    calling_point=calling_point_for_callback_fxn,
    observation_function=my_agent.observation_function,  # optional function
    actuation_function= my_agent.actuation_function,  # optional function
    update_state=True,  # use this callback to update the EMS state
    update_observation_frequency=1,  # linked to observation update
    update_actuation_frequency=1  # linked to actuation update
)

# -- RUN BUILDING SIMULATION --

max_episodes = 1

end_time = time.time()
print("Time for initialization: ", end_time - start_time, "s")

start_time = time.time()
for ep in range(max_episodes):
    sim.run_env(ep_weather_path)
    sim.reset_state() # reset when done

end_time = time.time()
print("Time for simulation: ", end_time - start_time, "s")

start_time = time.time()
# -- Sample Output Data --
output_dfs = sim.get_df(to_csv_file=cvs_output_path)  # LOOK at all the data collected here, custom DFs can be made too, possibility for creating a CSV file (GB in size)

# -- Plot Results --

dfa.plot_results()

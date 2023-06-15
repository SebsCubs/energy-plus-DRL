"""
Author: Sebastian Cubides

"""
import sys
sys.path.insert(0, '/usr/local/EnergyPlus-22-1-0')
from pyenergyplus import api #Importing from folder, therefore a warning may show
from pyenergyplus.api import EnergyPlusAPI
from emspy import EmsPy, BcaEnv
import datetime
import matplotlib.pyplot as plt
import tkinter
import data_frame_analysis as dfa


# -- FILE PATHS --
# * E+ Download Path *
ep_path = 'C:\EnergyPlusV22-1-0'  # path to E+ on system
# IDF File / Modification Paths
#idf_file_name = r'/home/jun/HVAC/repo_recover/energy-plus-DRL/BEMFiles/sdu_double_heating_dec_test.idf'  # building energy model (BEM) IDF file
idf_file_name = r'/home/jun/HVAC/repo_recover/energy-plus-DRL/BEMFiles/sdu_double_heating_dec_test.idf'
# Weather Path
ep_weather_path = r'/home/jun/HVAC/repo_recover/energy-plus-DRL/BEMFiles/DNK_Dec.epw'  # EPW weather file
# Output .csv Path (optional)
cvs_output_path = r'/home/jun/HVAC/repo_recover/energy-plus-DRL/Dataframes/dataframes_output_test.csv'


# STATE SPACE (& Auxiliary Simulation Data)

zn0 = 'Thermal Zone 1' #name of the zone to control 
tc_intvars = {}  # empty, don't need any

tc_vars = {
    # Building
    #'hvac_operation_sched': ('Schedule Value', 'HtgSetp 1'),  # is building 'open'/'close'?
    # -- Zone 0 (Core_Zn) --
    'zn0_temp': ('Zone Air Temperature', zn0),  # deg C
    'air_loop_fan_mass_flow_var' : ('Fan Air Mass Flow Rate','FANSYSTEMMODEL VAV'),  # kg/s
    'air_loop_fan_electric_power' : ('Fan Electricity Rate','FANSYSTEMMODEL VAV'),
    'damper_coil_heating_rate' : ('Heating Coil Heating Rate','Changeover Bypass HW Rht Coil'),  # W
    'pre_heating_coil_htgrate' : ('Heating Coil Heating Rate','HW Htg Coil'),
    'damper_mass_flow_var' : ('System Node Mass Flow Rate','CHANGEOVER BYPASS HW RHT DAMPER OUTLET NODE'),
    'vav_damper_position' : ('Zone Air Terminal VAV Damper Position','CHANGEOVER BYPASS HW RHT'),
    'vav_outdoor_flow_rate' : ('Zone Air Terminal Outdoor Air Volume Flow Rate','CHANGEOVER BYPASS HW RHT'),
    'ppd' : ('Zone Thermal Comfort Fanger Model PPD', 'THERMAL ZONE 1 189.1-2009 - OFFICE - WHOLEBUILDING - MD OFFICE - CZ4-8 PEOPLE'),
    'pmv' : ('Zone Thermal Comfort Fanger Model PMV', 'THERMAL ZONE 1 189.1-2009 - OFFICE - WHOLEBUILDING - MD OFFICE - CZ4-8 PEOPLE'),
    'deck_temp' : ('System Node Temperature','Node 30'),
    'post_deck_temp' : ('System Node Temperature','Node 13'),
    'total_hvac_energy' : ('Facility Total HVAC Electricity Demand Rate','WHOLE BUILDING'),
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
    #'zn0_cooling_sp': ('Zone Temperature Control', 'Cooling Setpoint', zn0),  # deg C
    #'zn0_heating_sp': ('Zone Temperature Control', 'Heating Setpoint', zn0),
    #'air_loop_fan_mass_flow_actuator' : ('Fan','Fan Air Mass Flow Rate','FanSystemModel ConstantVolume')  # kg/s
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



class Agent:
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
        self.time = None
        self.fan_mass_flow = None  # kg/s
        self.re_heating_vav_energy = None  # deg C
        self.pre_heating_coil_energy = None # deg C
        self.vav_mass_flow_rate = None # kg/s
        self.vav_heating_rate = None # W
        self.vav_damper_position = None # %
        self.vav_outdoor_flow_rate = None # m3/s
        self.ppd = None # %
        self.pmv = None # scale
        self.deck_temp = None # deg C
        self.post_deck_temp = None # deg C


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
            self.fan_mass_flow = var_data[1]
            self.re_heating_vav_energy = var_data[2]
            self.pre_heating_coil_energy = var_data[3]
            self.vav_mass_flow_rate = var_data[4]
            self.vav_damper_position = var_data[5]
            self.vav_outdoor_flow_rate = var_data[6]    
            self.ppd = var_data[7]
            self.pmv = var_data[8]
            self.deck_temp = var_data[9]
            self.post_deck_temp = var_data[10]

            # OR if using "return_dict=True"
            outdoor_temp = weather_data['oa_db']  # outdoor air dry bulb temp

            # print reporting
            """            
            if self.time.hour % 2 == 0 and self.time.minute == 0:  # report every 2 hours
                print(f'\n\nTime: {str(self.time)}')
                print('\n\t* Observation Function:')
                print(f'\t\tVars: {var_data}'  # outputs ordered list
                    f'\n\t\tWeather:{weather_data}')  # outputs dictionary
                print(f'\t\tZone0 Temp: {round(self.zn0_temp,2)} C')
                print(f'\t\tOutdoor Temp: {round(outdoor_temp, 2)} C')
            """
          

    def actuation_function(self):
        return 0


#  --- Create agent instance ---

my_agent = Agent(sim)



# --- Set your callback function (observation and/or actuation) function for a given calling point ---
sim.set_calling_point_and_callback_function(
    calling_point=calling_point_for_callback_fxn,
    observation_function=my_agent.observation_function,  # optional function
    actuation_function= None, #my_agent.actuation_function,  # optional function
    update_state=True,  # use this callback to update the EMS state
    update_observation_frequency=1,  # linked to observation update
    update_actuation_frequency=1  # linked to actuation update
)

# -- RUN BUILDING SIMULATION --

sim.run_env(ep_weather_path)
sim.reset_state()  # reset when done


   
# -- Sample Output Data --
output_dfs = sim.get_df(to_csv_file=cvs_output_path)  # LOOK at all the data collected here, custom DFs can be made too, possibility for creating a CSV file (GB in size)

# -- Plot Results --
fig, ax = plt.subplots()
output_dfs['var'].plot(y='zn0_temp', use_index=True, ax=ax)
#output_dfs['var'].plot(y='pmv', use_index=True, ax=ax)
plt.title('HVAC Electricity')
plt.show()
# Analyze results in "out" folder, DView, or directly from your Python variables and Pandas Dataframes

dfa.plot_only_baseline()
"""
Author: Sebastian Cubides
"""
import configparser
import os
import time
import pandas as pd
import numpy as np
from eplus_drl import EmsPy, BcaEnv
import datetime
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg') # For saving in a headless program. Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt


start_time = time.time()

# Load configuration
config = configparser.ConfigParser()
config.read('config.ini')

ep_path = config['DEFAULT']['ep_path']
idf_file_name = config['DEFAULT']['idf_file_name']
ep_weather_path = config['DEFAULT']['ep_weather_path']
cvs_output_path = config['DEFAULT']['cvs_output_path']
number_of_subprocesses = config.getint('DEFAULT', 'number_of_subprocesses')
number_of_episodes = config.getint('DEFAULT', 'number_of_episodes')
eplus_verbose = config.getint('DEFAULT', 'eplus_verbose')
state_size = tuple(map(int, config['DEFAULT']['state_size'].split(',')))
action_size = config.getint('DEFAULT', 'action_size')
learning_rate = config.getfloat('DEFAULT', 'learning_rate')
model_path = config['DEFAULT']['model_path']

# STATE SPACE (& Auxiliary Simulation Data)

zn0 = 'Thermal Zone 1' #name of the zone to control 
tc_intvars = {}  # empty, don't need any

tc_vars = {
    'zn0_temp': ('Zone Air Temperature', zn0),  # deg C
    'air_loop_fan_mass_flow_var' : ('Fan Air Mass Flow Rate','FANSYSTEMMODEL VAV'),  # kg/s
    'air_loop_fan_electric_power' : ('Fan Electricity Rate','FANSYSTEMMODEL VAV'),  # W
    'deck_temp_setpoint' : ('System Node Setpoint Temperature','Node 30'),  # deg C
    'deck_temp' : ('System Node Temperature','Node 30'),  # deg C
    'ppd' : ('Zone Thermal Comfort Fanger Model PPD', 'THERMAL ZONE 1 189.1-2009 - OFFICE - WHOLEBUILDING - MD OFFICE - CZ4-8 PEOPLE'),
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

class Energyplus_Agent:
    def __init__(self, bca: BcaEnv):
        self.bca = bca
        self.zn0_temp = None  # deg C
        self.fan_mass_flow = None  # kg/s
        self.fan_electric_power = None  # W
        self.deck_temp_setpoint = None  # deg C
        self.deck_temp = None  # deg C
        self.ppd = None  # percent
        self.time = None

        # -- RL AGENT --
        self.state_size = state_size
        self.action_size = action_size

        self.Actor = self.load(model_path) 
        self.state_window = 3 
        self.states = []
        self.not_averaged_state = []
        self.state = None
        self.time_of_day = None

    def observation_function(self):
        self.time = self.bca.get_ems_data(['t_datetimes'])
        if self.time < datetime.datetime.now():
            var_data = self.bca.get_ems_data(list(self.bca.tc_var.keys()))
            weather_data = self.bca.get_ems_data(list(self.bca.tc_weather.keys()), return_dict=True)

            self.zn0_temp = var_data[0]  
            self.fan_mass_flow = var_data[1]  # kg/s
            self.fan_electric_power = var_data[2]  # W
            self.deck_temp_setpoint = var_data[3]  # deg C
            self.deck_temp = var_data[4]  # deg C
            self.ppd = var_data[5]  # percent

            self.state = self.get_state(var_data, weather_data)
            self.states.append(self.state)
                  
    def actuation_function(self):        
        if self.time < datetime.datetime.now():
            action = self.act(self.state)    
            fan_flow_rate = action * (2.18 / 10)
        return { 'fan_mass_flow_act': fan_flow_rate }

    def get_state(self, var_data, weather_data):   
        self.time_of_day = self.bca.get_ems_data(['t_hours'])
        state = np.concatenate((np.array([self.time_of_day]), var_data[:6], list(weather_data.values())[:2])) 

        state[0] = state[0] / 24
        state[1] = (state[1] - 15) / 20
        state[2] = state[2] / 2.18
        state[3] = state[3] / 3045.81
        state[4] = (state[4] - 15) / 15
        state[5] = state[5] / 35
        state[6] = state[6] / 100
        state[7] = state[7] / 100
        state[8] = (state[8] + 10) / 20
        return state

    def act(self, state):
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_probs, _ = self.Actor(state)
        action = np.random.choice(self.action_size, p=action_probs.numpy().squeeze())
        return action
        
    def load(self, Actor_name):
        model = ActorCriticModel(self.state_size, self.action_size)
        model.load_state_dict(torch.load(Actor_name))
        model.eval()
        return model

#  --- Create agent instance ---
my_agent = Energyplus_Agent(sim)

# --- Set your callback function (observation and/or actuation) function for a given calling point ---
sim.set_calling_point_and_callback_function(
    calling_point=calling_point_for_callback_fxn,
    observation_function=my_agent.observation_function,  # optional function
    actuation_function=my_agent.actuation_function,  # optional function
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
    sim.reset_state()  # reset when done

end_time = time.time()
print("Time for simulation: ", end_time - start_time, "s")

start_time = time.time()
# -- Sample Output Data --
output_dfs = sim.get_df(to_csv_file=cvs_output_path)  # LOOK at all the data collected here, custom DFs can be made too, possibility for creating a CSV file (GB in size)

def plot_results():
    # Check if the Figures/ directory exists and create it if not
    figures_dir = 'Figures'
    if not os.path.exists(figures_dir):
        os.makedirs(figures_dir)

    baseline_dfs = pd.read_csv("Dataframes/dataframes_output_model.csv")
    output_dfs = pd.read_csv("Dataframes/dataframes_output_test.csv")
    
    ########## Temperature ##########

    y_axis = output_dfs['zn0_temp']
    y_baseline = baseline_dfs['zn0_temp']
    fig, ax = plt.subplots()
    vplot1 = ax.violinplot(y_axis, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax.violinplot(y_baseline, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)

    ax.set_title('Room temperature over December (Temperature reward, ext. Fan)')
    ax.set_xlabel('Degrees (Celsius)')
    ax.set_ylabel('Density')
    temp_model_mean = np.mean(y_axis)
    temp_baseline_mean = np.mean(y_baseline)
    temp_model_label = f"{'RL Agent (mean: '}{temp_model_mean:.2f}{')'}"
    temp_baseline_labe = f"{'Baseline controller (mean: '}{temp_baseline_mean:.2f}{')'}"
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[temp_model_label, temp_baseline_labe])
    #save figure in a folder called "Figures"
    plt.savefig('Figures/room_temp.png', dpi=300, bbox_inches="tight")

    ########## PPD ##########
    y_axis = output_dfs['ppd']
    y_baseline = baseline_dfs['ppd']
    fig, ax = plt.subplots()
    vplot1 = ax.violinplot(y_axis, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax.violinplot(y_baseline, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)

    ax.set_title('PPD over December (Temperature reward, ext. Fan)')
    ax.set_xlabel('PPD (percentage)')
    ax.set_ylabel('Density')
    ppd_model_mean = np.mean(y_axis)
    ppd_baseline_mean = np.mean(y_baseline)
    ppd_model_label = f"{'RL Agent (mean: '}{ppd_model_mean:.2f}{')'}"
    ppd_baseline_labe = f"{'Baseline controller (mean: '}{ppd_baseline_mean:.2f}{')'}"
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[ppd_model_label, ppd_baseline_labe])
    #save figure in a folder called "Figures"
    plt.savefig('Figures/ppd.png', dpi=300, bbox_inches="tight")

        ########## Power usage ##########

    y_axis = output_dfs['air_loop_fan_electric_power']
    y_baseline = baseline_dfs['air_loop_fan_electric_power']
    fig, ax = plt.subplots()
    vplot1 = ax.violinplot(y_axis, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)
    vplot2 = ax.violinplot(y_baseline, points=200, vert=False, widths=1.1,
                        showmeans=True, showextrema=True, showmedians=False,
                        bw_method=0.5)

    ax.set_title('Fan power usage over December (Temperature reward, ext. Fan)')
    ax.set_xlabel('Power (Watts)')
    ax.set_ylabel('Density')
    power_model_mean = np.mean(y_axis)
    power_baseline_mean = np.mean(y_baseline)
    power_model_label = f"{'RL Agent (mean: '}{power_model_mean:.2f}{')'}"
    power_baseline_labe = f"{'Baseline controller (mean: '}{power_baseline_mean:.2f}{')'}"
    ax.legend(handles=[vplot1["bodies"][0], vplot2["bodies"][0]], labels=[power_model_label, power_baseline_labe])
    #save figure in a folder called "Figures"
    plt.savefig('Figures/fan_power.png', dpi=300, bbox_inches="tight")


    ######### Room temperature time series #########

    # -- Plot Results --
    #plot with date in the x-axis
    fig, ax = plt.subplots()
    output_dfs.plot(x='Datetime', y='zn0_temp', use_index=True, ax=ax)
    #Reduce the number of x-axis labels
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    output_dfs.plot(x='Datetime', y='oa_db', use_index=True, ax=ax)
    plt.title('Room temperature (Time)')
    
    #save figure in a folder called "Figures"
    plt.savefig('Figures/room_temp_time_series.png', dpi=300, bbox_inches="tight")

plot_results()

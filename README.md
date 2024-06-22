# eplus_DRL

### This library was created to allow for quick prototyping, training and testing of building control strategies based on Deep Reinforcement Learning (DRL). It was initially inspired by mechyai's [RL-EmsPy library](https://github.com/mechyai/RL-EmsPy). I attempted to provide a more streamlined process to create different control strategies, enable parallelization, plotting, versioning and a more complete release in general.

*Any feedback or improvements to the repo are welcome.* 

## Getting started
1. Requirements
2. Installation
3. How to use
4. Library's roadmap

## 1. Requirements
- A Linux-based OS (Running multiple instances of energyplus requires multiprocessing since running it in a single process exhausts the memory due to memory handles not closing after an energyplus run)
- Python 3.10.4
- EnergyPlus v >= 22.1.0 installed in your system, keep in mind the installation directory to modify the config files.
- (Optional, strongly recommended) a python virtual environment

## 2. Installation
With conda/miniconda installed and an open terminal:
```
conda create --name eplus_env python=3.10.4
```
```
conda activate eplus_env
```
```
git clone https://github.com/SebsCubs/eplus_drl.git
```
```
cd eplus_drl
```
Make sure you have the virtualenv active:
```
pip install -e eplus_drl/
```

## 3. How to use

Here I'll show with an example a typical workflow for using the library:

1. Brainstorm ideas for controlling building systems, establish key performance indicators for your building simulation
2. Run your energyplus model once, fetch available variables and actuators
3. Decide which variables to observe from the energyplus simulation, decide how to normalize/embed them
4. Decide which actuators to use and a reward function if applicable
5. Decide the agent's architecture and network hyperparameters
6. Train the agent's policy
7. Test the agent
8. Visualize and compare the new controller's impact on the building simulation
9. Benchmark different control strategies

### 1. Brainstorm ideas for controlling building systems, establish key performance indications for the building simulation
There are infinite opportunities for controlling building energy systems. This is also a big part of the motivation for the research in this fleid. It is a clear example of the so called "curse of dimensionality" where there are many control variables and multiple possible references and desired behaviors. Creativity needs to be used to create streamlined building control systems and higher levels of automation for these systems. In this example I am creating a RL-agent to control the ventilation system of a room in an office building:

![image](https://github.com/SebsCubs/eplus_drl/assets/35004035/50506445-7f59-4af4-b7ed-f5065d12a359)


Now, we need to define a control strategy to guide the rest of the development process. In this example, I will monitor the temperature of one room (Thermal Zone 1) and then adjust the main fan's air mass flow rate accordingly.

While this isn't a realistic control strategy—since controlling temperature usually involves adjusting the heating coil's heating rate as well—it serves our example purposes. Additionally, EnergyPlus currently lacks an actuator for heating coils.

As a general rule, it is preferable to use setpoints as actuators since they are more widely available. However, this is case-dependent and requires some experience to develop a mental model of the controlling capabilities offered by EnergyPlus. Not every control strategy you can conceive is feasible with EnergyPlus.

![image](https://github.com/SebsCubs/eplus_drl/assets/35004035/3f51f2d7-83ff-4650-aa00-b5ee5cb80840)



### 2. Get the list of available variables and actuators
[EnergyPlus](https://energyplus.net/) has different files resulting from a simulation. Two files in specific contain all the observable variables and the possible actuators that can be used in any given Energyplus model: eplusout.edd for actuators and eplusout.rdd for variables (also eplusout.mdd for meter but I have never used them). To obtain this files, the model must be run once, making sure that the .idf file contains the verbose option to output the .edd file. Make sure this object exists in the .idf file:
```
Output:EnergyManagementSystem,
    Verbose,    ! Actuator Availability Dictionary Reporting
    Verbose,    ! Internal Variable Availability Dictionary Reporting
    ErrorsOnly; ! EnergyPlus Runtime Language Debug Output Level
```
Here I'm running it from WSL from a windows machine:

![image](https://github.com/SebsCubs/eplus_drl/assets/35004035/66b2c97e-931a-4377-9397-29d33fe8b226)

And I run the Office_building_Copenhagen example
```
conda activate eplus_env
```
```
cd eplus_drl/examples/rl_ventilation_control/Office_building_Copenhagen
```
```
python no_actuation_simulation.py
```

A folder named "out" should now be available in the example folder, inside the .rdd and .edd files should be available. 

Then after exploring the .edd and .rdd files I decided to observe these variables, extra to the ones already set in the .idf file. (I just copied and pasted these variables from the .rdd file)

```
Output:Variable,*,Heating Coil Heating Rate,hourly; !- HVAC Average [W]
Output:Variable,*,System Node Mass Flow Rate,hourly; !- HVAC Average [kg/s]
Output:Variable,*,Zone Air Terminal VAV Damper Position,hourly; !- HVAC Average []
Output:Variable,*,Zone Air Terminal Minimum Air Flow Fraction,hourly; !- HVAC Average []
Output:Variable,*,Zone Air Terminal Outdoor Air Volume Flow Rate,hourly; !- HVAC Average [m3/s]
Output:Variable,*,Zone Thermal Comfort Fanger Model PMV,hourly; !- Zone Average []
Output:Variable,*,Zone Thermal Comfort Fanger Model PPD,hourly; !- Zone Average [%]
Output:Variable,*,Pump Electricity Rate,hourly; !- HVAC Average [W]
Output:Variable,*,Facility Total HVAC Electricity Demand Rate,hourly; !- HVAC Average [W]
```
For the actuators you don't need to modify the .idf file. Just use them in the eplus_drl dictionaries.

### 3. Design of an RL-Agent for the control-strategy

Design of reinforcement learning agents is a vast topic, I will just cover the general steps on how to do it for this kind of setup:
1. Decide a reinforcement learning algorithm and cost functions: I am using a parallelized version of A2C here, I tried following [openAI's baseline blogpost](https://openai.com/index/openai-baselines-acktr-a2c/)
2. Define agent's policy network (likely a Deep Neural Network): I have used 4 layers with 512 nodes, fully connected and 0.001 learning rate. See the [policy.py](https://github.com/SebsCubs/eplus_drl/blob/main/examples/rl_ventilation_control/Parallel_A2C/policy.py) file.
3. Define agent's input/output and data embeddings (normalization):
The state of the agent is this:
  ![image](https://github.com/SebsCubs/eplus_drl/assets/35004035/ece62480-309b-45bb-a6fb-ca87d06ac40e)

Where I normalize all the variables with min/max normalization after monitoring a couple of simulations and deciding min/max values for each.

The output (action) of the agent is:

![image](https://github.com/SebsCubs/eplus_drl/assets/35004035/59542882-c9f8-4992-ad6e-c2ba00040737)

Where m_fan is the air mass flow rate of the main fan, a agent is the network's output, an integer between 0 and 10.


5. Design and test a reward function:
   
![image](https://github.com/SebsCubs/eplus_drl/assets/35004035/91bc7591-fe08-4f41-9920-2c66ee2970de)

Where SP1 is the room's temperature setpoint, T_indoor is the actual room's temperature and P_fan is the fan's power consumption. Effectively trying to minimize the difference between reference and actual room temperature while also minimizing the fan's electricity consumption.

The implementation of all of this concepts can be found in the example folder: [ParallelA2C](https://github.com/SebsCubs/eplus_drl/tree/main/examples/rl_ventilation_control/Parallel_A2C)
Where
- [main.py](https://github.com/SebsCubs/eplus_drl/blob/main/examples/rl_ventilation_control/Parallel_A2C/main.py) is the main file to run the example, it contains the gist of the multiprocessing implementation.
- [policy.py](https://github.com/SebsCubs/eplus_drl/blob/main/examples/rl_ventilation_control/Parallel_A2C/policy.py) contains the agent's network (actor and critic) and the act() method
- [eplus_manager.py](https://github.com/SebsCubs/eplus_drl/blob/main/examples/rl_ventilation_control/Parallel_A2C/eplus_manager.py) contains the energyplus environment to run simulations and harvest experience data from the model interactions to train the global policy it contains the reward function.
- [a2c.py](https://github.com/SebsCubs/eplus_drl/blob/main/examples/rl_ventilation_control/Parallel_A2C/a2c.py) contains the implementation of the A2C algorithm and is the instance containing the global policy. It receives the experience data and runs training.

6. Train and testing

Agent is still training :-) 

## 4. Library's roadmap

- Create a comprehensive test strategy and cover the codebase. Create CI-CD pipelines with github actions
- A ReadTheDocs (or github pages) documentation website.
- Develop a .idf file (energyplus' model file) utility to modify it without having to do manual changes to adapt it for use with this library.
- Wrap the library to make it Pypi-ready
- A [Neuromancer](https://github.com/pnnl/neuromancer) dependency was added, which is *an open-source differentiable programming (DP) library for solving parametric constrained optimization problems, physics-informed system identification, and parametric model-based optimal control.* Implementing control strategies using this library (e.g. Differentiable MPC) and provide some examples is an interesting exploration.
- Implementing different Reinforcement Learning algorithms to compare their effectiveness, like PPO and ACKTR.
- Create an abstract model to provide a framework for creating building control strategies using model predictive control (MPC), and ML-based control and modeling techniques.
- Feel free to contribute with any of these or any other functionalities you find interesting ;) 

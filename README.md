# eplus-DRL

### This library was created to allow for quick prototyping, training and testing of building control strategies based on Deep Reinforcement Learning (DRL). It is heavily inspired by mechyai's [RL-EmsPy library](https://github.com/mechyai/RL-EmsPy). I attempted to provide a more streamlined process to create different control strategies, enable parallelization, plotting, versioning and a more complete release in general.

*Any feedback or improvements to the repo are welcome.* 

## Getting started
1. Requirements
2. Installation
3. Brainstorm ideas for controlling building systems, establish key performance indicators for your building simulation
4. Run your energyplus model once, fetch available variables and actuators
5. Decide which variables to observe from the energyplus simulation, decide how to normalize/embed them
6. Decide which actuators to use and a reward function if applicable
7. Decide the agent's architecture and network hyperparameters
8. Train the agent's policy
9. Test the agent
11. Visualize and compare the new controller's impact on the building simulation
12. Benchmark different control strategies

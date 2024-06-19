import time
from utils import load_config
from eplus_manager import Energyplus_manager
from policy import Policy
from a2c import A2C_trainer
from multiprocessing import Pool
import multiprocessing as mp
import os
import copy



def run_eplus_experience_harvesting(queue, episode, global_policy, config):
    while queue.qsize() >= config['max_queue_size']:
        time.sleep(0.1)  # Wait for the queue to have space
    control_policy = copy.deepcopy(global_policy)
    eplus_object = Energyplus_manager(episode, control_policy, config)
    eplus_object.run_episode()
    return eplus_object.episode_reward

def global_policy_process(queue, a2c_object):
    max_number_of_episodes = a2c_object.config['max_number_of_episodes']
    
    while True:
        experience_batch = queue.get()
        
        # Assuming experience_batch has an attribute 'episode' that stores the episode number
        if experience_batch.episode >= max_number_of_episodes:
            print(f"Reached max number of episodes: {experience_batch.episode}")
            break
        
        # Update the global policy with the experience batch
        a2c_object.update(experience_batch)

    



def main():
    pid = os.getpid()
    print("Main process, pid: ", pid)

    config = load_config()
    pool_size = config['number_of_subprocesses'] 
    EPISODES = config['number_of_episodes']

    experience_queue = mp.Queue()
    global_policy = Policy(config['state_size'], config['action_size'])
    a2c_object = A2C_trainer(global_policy,config)
    

    with Pool(processes=pool_size, maxtasksperchild=3) as pool:
        pool.apply_async(global_policy_process, args=(experience_queue, a2c_object))
        for index in range(EPISODES):
            pool.apply_async(run_eplus_experience_harvesting, args=(experience_queue, index, global_policy, config))
        pool.close()
        pool.join()



if __name__ == "__main__":
    main()  
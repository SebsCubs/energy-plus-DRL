import time
import os
import copy
import logging
from multiprocessing import Pool, Manager
from utils import load_config
from eplus_manager import Energyplus_manager
from policy import Policy
from a2c import A2C_trainer

def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        handlers=[
            logging.FileHandler("a2c_example_program.log"),
            logging.StreamHandler()
        ]
    )

def run_eplus_experience_harvesting(queue, episode, global_policy, config):
    pid = os.getpid()
    logging.info(f"Experience harvesting process number: {episode}, pid: {pid}")
    try:
        while True:
            if queue.qsize() < config['max_queue_size']:
                break
            time.sleep(0.1)  # Wait for the queue to have space

        control_policy = copy.deepcopy(global_policy)

        eplus_object = Energyplus_manager(episode, control_policy, config)
        eplus_object.run_episode()

        queue.put(eplus_object.episode_reward)

        logging.info(f"Episode {episode} completed with reward: {eplus_object.episode_reward}")
    except Exception as e:
        logging.error(f"Error in episode {episode}: {e}")

def global_policy_process(queue, a2c_object):
    pid = os.getpid()
    logging.info(f"Global policy process, pid: {pid}")
    max_number_of_episodes = a2c_object.config['max_number_of_episodes']
    
    while True:
        try:
            experience_batch = queue.get(timeout=5)  # timeout to allow periodic checks
        except queue.Empty:
            continue  # No message, loop again
        
        try:
            # Assuming experience_batch has an attribute 'episode' that stores the episode number
            if experience_batch.episode >= max_number_of_episodes:
                logging.info(f"Reached max number of episodes: {experience_batch.episode}")
                break
            
            # Update the global policy with the experience batch
            a2c_object.update(experience_batch)
        except Exception as e:
            logging.error(f"Error processing experience batch: {e}")
            # Optionally, you can choose to break the loop or handle the error differently

    logging.info("Shutting down global policy process")

def main():
    pid = os.getpid()

    setup_logging()
    logging.info(f"Main process, pid: {pid}")
    print(f"Main process, pid: {pid}", flush=True)

    config = load_config()
    pool_size = config['number_of_subprocesses']
    EPISODES = config['number_of_episodes']

    manager = Manager()
    experience_queue = manager.Queue()
    global_policy = Policy(config['state_size'], config['action_size'])
    a2c_object = A2C_trainer(global_policy, config)

    with Pool(processes=pool_size, maxtasksperchild=3) as pool:
        results = []
        
        print("Starting global policy process", flush=True)
        logging.info("Starting global policy process")
        result = pool.apply_async(global_policy_process, args=(experience_queue, a2c_object))
        results.append(result)

        print("Starting experience harvesting processes", flush=True)
        logging.info("Starting experience harvesting processes")
        for index in range(EPISODES):
            result = pool.apply_async(run_eplus_experience_harvesting, args=(experience_queue, index, global_policy, config))
            results.append(result)

        pool.close()
        pool.join()

        # Ensure all tasks have completed
        for result in results:
            result.get()  # This will raise exceptions if any occurred during execution

    logging.info("All subprocesses have completed.")
    print("All subprocesses have completed.", flush=True)

if __name__ == "__main__":
    setup_logging()
    main()

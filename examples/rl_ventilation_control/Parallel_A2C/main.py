import time
import os
import copy
import logging
from multiprocessing import Pool, Manager, queues
from eplus_drl.utils import load_config
from eplus_manager import Energyplus_manager
from policy import Policy
from a2c import A2C_trainer
import numpy as np

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
    logging.debug(f"Experience harvesting process number: {episode}, pid: {pid}")
    try:
        while True:
            if queue.qsize() < config['queue_size_max']:
                break
            time.sleep(0.1)  # Wait for the queue to have space

        control_policy = copy.deepcopy(global_policy)

        eplus_object = Energyplus_manager(episode, control_policy, config)
        eplus_object.run_episode()

        episode_experience = {
            'episode': episode,
            'states': eplus_object.states,
            'actions': eplus_object.actions,
            'rewards': eplus_object.rewards
        }

        queue.put(episode_experience)

        logging.debug(f"Episode {episode} completed with reward: {np.sum(eplus_object.rewards)}")
    except Exception as e:
        logging.error(f"Error in episode {episode}: {e}")


def global_policy_process(queue, a2c_object):
    pid = os.getpid()
    logging.debug(f"Global policy process, pid: {pid}")
    max_number_of_episodes = a2c_object.config['number_of_episodes']
    
    while True:
        try:
            experience_batch = queue.get(timeout=5)  # timeout to allow periodic checks
        except queues.Empty:
            continue  # No message, loop again
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            continue

        try:
            if experience_batch['episode'] >= max_number_of_episodes:
                logging.info(f"Reached max number of episodes: {experience_batch['episode']}")
                break
            
            # Update the global policy with the experience batch
            a2c_object.update(experience_batch)
            
        except Exception as e:
            logging.error(f"Error processing experience batch: {e}")

    logging.info("Shutting down global policy process")


def main():
    pid = os.getpid()

    setup_logging()
    logging.info(f"Main process, pid: {pid}")

    config = load_config()
    pool_size = config['number_of_subprocesses']
    EPISODES = config['number_of_episodes']

    manager = Manager()
    experience_queue = manager.Queue()
    global_policy = Policy(config['state_size'], config['action_size'])
    a2c_object = A2C_trainer(global_policy, config)

    with Pool(processes=pool_size, maxtasksperchild=3) as pool:
        results = []
        logging.info("Starting global policy process")
        result = pool.apply_async(global_policy_process, args=(experience_queue, a2c_object))
        results.append(result)

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

if __name__ == "__main__":
    setup_logging()
    main()

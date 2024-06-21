import configparser

def load_config(config_file='config.ini'):
    config = configparser.ConfigParser()
    config.read(config_file)
    
    config_dict = {
        'ep_path': config['DEFAULT']['ep_path'],
        'idf_file_name': config['DEFAULT']['idf_file_name'],
        'ep_weather_path': config['DEFAULT']['ep_weather_path'],
        'cvs_output_path': config['DEFAULT']['cvs_output_path'],
        'number_of_subprocesses': config.getint('DEFAULT', 'number_of_subprocesses'),
        'number_of_episodes': config.getint('DEFAULT', 'number_of_episodes'),
        'eplus_verbose': config.getint('DEFAULT', 'eplus_verbose'),
        'state_size': tuple(map(int, config['DEFAULT']['state_size'].split(','))),
        'action_size': config.getint('DEFAULT', 'action_size'),
        'learning_rate': config.getfloat('DEFAULT', 'learning_rate'),
        'model_path': config['DEFAULT']['model_path'],
        'queue_size_max' : config.getint('DEFAULT', 'queue_size_max')
    }
    
    return config_dict
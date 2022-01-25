import argparse
import datetime
import shutil
from pathlib import Path

import numpy as np
import yaml

from predictive_hands.data_loading.DataContainer import DataContainer


def createConfigTest(cfg):
    s_config = cfg.copy()
    models_dir = Path(cfg['experiment_dir']).joinpath('models')
    results_dir = Path(cfg['experiment_dir']).joinpath('results')
    config_dir = Path(cfg['experiment_dir']).joinpath('config')
    if Path(cfg['experiment_dir']).exists():
        shutil.rmtree(Path(cfg['experiment_dir']), ignore_errors=True)

    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)
    s_config['models_dir'] = str(models_dir)
    s_config['results_dir'] = str(results_dir)
    s_config['config_dir'] = str(config_dir)

    yml_out_path = config_dir.joinpath('config').with_suffix(".yml")
    yaml.dump(s_config, open(yml_out_path,'w'))

def runBaseline(folder_name):
    yml = Path(folder_name).joinpath('config', 'config').with_suffix('.yml')
    cfg = yaml.full_load(open(yml,'r'))
    DataContainer.generateBaseline(cfg)


experiment_type_dict = {'config_exp': createConfigTest}

if __name__ == "__main__":

    msg = ''' 
        You know, some stuff at some point.
            '''

    parser = argparse.ArgumentParser(description='Train prediction')
    parser.add_argument('--directories_file', required=True, type=str,
                        help='Path to directories for given machine')
    parser.add_argument('--config_file', required=False, type=str,
                        help='Path to config file')
    parser.add_argument('--experiment_name', required=False, type=str,
                        help='Name of experiment (optional if provided in config)')

    args, extras = parser.parse_known_args()

    current_config= args.config_file
    experiment_name = args.experiment_name

    # Extras is a list in the form ["--arg1", "val1", "--arg2", "val2"]. Convert it to a dictionary
    parser_dict = {extras[i].replace('--', ''): yaml.full_load(extras[i + 1]) for i in range(0, len(extras), 2)}

    if current_config is None:
        current_config = 'configs/default_config.yml'

    directories_cfg = yaml.full_load(open(args.directories_file, 'r'))
    cfg = yaml.full_load(open(current_config,'r'))

    cfg.update(directories_cfg)

    if experiment_name is not None:
        cfg['experiment_name'] = experiment_name

    cfg['experiment_dir'] = str(Path(cfg['exp_dir_base']).joinpath('experiments', cfg['experiment_name']))

    cfg['datetime'] = str(datetime.datetime.now())
    for key in parser_dict.keys():
        cfg[key] = parser_dict[key]
    new_test_ranges = []
    if len(cfg['times_ahead']) == 1:
        cfg['times_ahead'] = list(range(cfg['times_ahead'][0]+1))
    for test_range in cfg['test_ranges']:
        if test_range <= max(cfg['times_ahead']):
            new_test_ranges.append(test_range)
    cfg['test_ranges'] = new_test_ranges

    if cfg['joint_noise_error'] is not None:
        cfg['joint_noise'] = cfg['joint_noise_error']*(np.sqrt(np.pi/8))
        cfg['transl_noise'] = cfg['transl_noise_error'] * (np.sqrt(np.pi / 8))
    else:
        cfg['transl_noise'] = None
        cfg['joint_noise'] = None

    createConfigTest(cfg)
    runBaseline(cfg['experiment_dir'])

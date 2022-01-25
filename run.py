import argparse
import datetime
import shutil
from pathlib import Path

import yaml

from predictive_hands.train.train_loop import run_training


def createConfigTest(cfg):
    s_config = cfg.copy()
    models_dir = Path(cfg['experiment_dir']).joinpath('models')
    results_dir = Path(cfg['experiment_dir']).joinpath('results')
    config_dir = Path(cfg['experiment_dir']).joinpath('config')
    if not cfg['continue_experiment']:
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

def runExperiment(folder_name):
    yml = Path(folder_name).joinpath('config', 'config').with_suffix('.yml')
    run_training(yml)
    return

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
    print(args)
    print(extras)
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

    if cfg['continue_experiment'] and Path(cfg['experiment_dir']).exists():
        # if we want to continue experiment and it exists, don't touch cfg file
        # this means if you name two experiments the same, it'll continue the old one
        # But do add anything explicitly flagged
        cfg_old_name = Path(cfg['experiment_dir']).joinpath('config', 'config').with_suffix('.yml')
        cfg_old = yaml.full_load(open(cfg_old_name, 'r'))
        for key in parser_dict.keys():
            cfg_old[key] = parser_dict[key]
        directories_cfg = yaml.full_load(open(args.directories_file, 'r'))
        cfg_old.update(directories_cfg)
        yaml.dump(cfg_old, open(cfg_old_name, 'w'))
    else:
        cfg['datetime'] = str(datetime.datetime.now())
        for key in parser_dict.keys():
            cfg[key] = parser_dict[key]
        if cfg['joint_noise_level'] is not None or cfg['transl_noise_level'] is not None:
            if cfg['joint_noise_level'] is None:
                cfg['joint_noise_level'] = 0
            if cfg['transl_noise_level'] is None:
                cfg['transl_noise_level'] = 0
            cfg['data_path'] += f'_{cfg["joint_noise_level"]}_{cfg["transl_noise_level"]}_noise4'
        new_test_ranges = []
        if len(cfg['times_ahead']) == 1 and not cfg['single_times']:
            cfg['times_ahead'] = list(range(cfg['times_ahead'][0]+1))
        for test_range in cfg['test_ranges']:
            if test_range <= max(cfg['times_ahead']):
                new_test_ranges.append(test_range)
        cfg['test_ranges'] = new_test_ranges
        experiment_type_dict[cfg['experiment_type']](cfg)
    runExperiment(cfg['experiment_dir'])

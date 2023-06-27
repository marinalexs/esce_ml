from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

from predefined_datasets import predefined_datasets as PREDEFINED_DATASETS
from models import MODELS
from base_models import RegressionModel
import os

def validate_root(config):
    errors = []
    
    # check that the config has the expected keys
    for key in ['val_test_frac', 'bootstrap_repetitions', 'seeds', 'sample_sizes', 'experiments', 'custom_datasets']:
        if key not in config:
            errors.append(f"config is missing expected key: {key}")
    
    # check that the config keys are of the expected type
    if not isinstance(config['val_test_frac'], float):
        errors.append(f"config['val_test_frac'] should be a float")
    if config['val_test_frac'] <= 0 or config['val_test_frac'] > 1:
        errors.append(f"config['val_test_frac'] should be between 0 and 1")
    if not isinstance(config['bootstrap_repetitions'], int):
        errors.append(f"config['bootstrap_repetitions'] should be an int")
    if config['bootstrap_repetitions'] <= 0:
        errors.append(f"config['bootstrap_repetitions'] should be greater than 0")
    if not isinstance(config['seeds'], list):
        errors.append(f"config['seeds'] should be a list")
    if not isinstance(config['sample_sizes'], list):
        errors.append(f"config['sample_sizes'] should be a list")
    if not isinstance(config['experiments'], dict):
        errors.append(f"config['experiments'] should be a dict")
    if not isinstance(config['custom_datasets'], dict):
        errors.append(f"config['custom_datasets'] should be a dict")
    
    # check that the config keys have the expected values
    for seed in config['seeds']:
        if not isinstance(seed, int):
            errors.append(f"config['seeds'] should be a list of ints")
        if seed < 0:
            errors.append(f"config['seeds'] should be a list of positive (!) ints")
    for sample_size in config['sample_sizes']:
        if not isinstance(sample_size, int):
            errors.append(f"config['sample_sizes'] should be a list of ints")
        if sample_size < 1:
            errors.append(f"config['sample_sizes'] should be a list of positive (!) ints")

    return errors


def validate_experiments(config):
    errors = []
    
    # check that each experiment has the expected keys
    for exp_name, exp in config['experiments'].items():
        for key in ['dataset', 'features', 'features_cni', 'targets', 'targets_cni', 'matching', 'models', 'grid']:
            if key not in exp:
                errors.append(f"experiment {exp_name} is missing expected key: {key}")
    
    # check that each experiment has the expected values
    for exp_name, exp in config['experiments'].items():
        if not isinstance(exp['features'], list):
            errors.append(f"experiment {exp_name}['features'] should be a list")
        if not isinstance(exp['features_cni'], list):
            errors.append(f"experiment {exp_name}['features_cni'] should be a list")
        if not isinstance(exp['targets'], list):
            errors.append(f"experiment {exp_name}['targets'] should be a list")
        if not isinstance(exp['targets_cni'], list):
            errors.append(f"experiment {exp_name}['targets_cni'] should be a list")
        if not isinstance(exp['matching'], list):
            errors.append(f"experiment {exp_name}['matching'] should be a list")
        if not isinstance(exp['models'], list):
            errors.append(f"experiment {exp_name}['models'] should be a list")
        if not isinstance(exp['grid'], str):
            errors.append(f"experiment {exp_name}['grid'] should be a string")
        
        # check that dataset exists
        if exp['dataset'] not in PREDEFINED_DATASETS and exp['dataset'] not in config['custom_datasets']:
            errors.append(f"experiment {exp_name}['dataset'] should be a string from: {PREDEFINED_DATASETS.keys()} or {config['custom_datasets'].keys()}")
        else:
            # check that all features, targets, covariates exist
            dataset = PREDEFINED_DATASETS[exp['dataset']] if exp['dataset'] in PREDEFINED_DATASETS else config['custom_datasets'][exp['dataset']]
            for feature in exp['features']:
                if feature not in dataset['features']:
                    errors.append(f"experiment {exp_name} feature {feature} must exist in either predefined  or custom datasets")
            for feature_cni in exp['features_cni']:
                if feature_cni != 'none' and feature_cni not in dataset['covariates']:
                    errors.append(f"experiment {exp_name} feature_cni {feature_cni} must exist in either predefined  or custom datasets")
            for target in exp['targets']:
                if target not in dataset['targets']:
                    errors.append(f"experiment {exp_name} target {target} must exist in either predefined  or custom datasets")
            for target_cni in exp['targets_cni']:
                if target_cni != "none" and target_cni not in dataset['covariates']:
                    errors.append(f"experiment {exp_name} targets_cni {target_cni} must exist in either predefined  or custom datasets")
            for  matching in exp['matching']:
                if matching not in ['none','balanced'] and matching not in dataset['covariates']:
                    errors.append(f"experiment {exp_name} matching {matching} must exist in either predefined or custom datasets")

        # check that model exists in MODELS
        for model in exp['models']:
            if not isinstance(model, str):
                errors.append(f"experiment {exp_name}['models'] should be a list of strings")
            if model not in MODELS:
                errors.append(f"experiment {exp_name}['models'] should be a list of strings from: {MODELS}")

        # check that grid has a corresponding yaml file in config/grids
        # and that all models have a key in the grid
        grid_path = os.path.join('config', 'grids', exp['grid'] + '.yaml')
        if not os.path.exists(grid_path):
            errors.append(f"experiment {exp_name}['grid'] should be a string from: {os.listdir('config/grids')}")
        else:
            with open(grid_path, 'r') as f:
                grid = yaml.load(f)
            for model in exp['models']:
                if model not in grid:
                    errors.append(f"experiment {exp_name}['grid'] should have hyperparameters defined for model: {model}")

        # if target_cni is not none, then matching must be none and model must be regression
        if exp['targets_cni'] is not ['none']:
            if exp['matching'] is not ['none']:
                errors.append(f"experiment {exp_name} matching must be 'none' when {exp_name} targets_cni is not 'none'")
            for model in exp['models']:
                if not isinstance(MODELS[model], RegressionModel):
                    errors.append(f"experiment {exp_name} model {model} must be regression model when {exp_name} targets_cni is not 'none'")

    return errors

def validate_custom_datasets(config):
    errors = []
    
    # check that each dataset has the expected keys
    for dataset_name, dataset in config['custom_datasets'].items():
        for key in ['features', 'targets', 'covariates']:
            if key not in dataset:
                errors.append(f"dataset {dataset_name} is missing expected key: {key}")
        if '_' in dataset_name:
            errors.append(f"dataset {dataset_name} should be string without underscores")

    
    # check that each dataset has the expected values
    for dataset_name, dataset in config['custom_datasets'].items():
        if not isinstance(dataset['features'], dict):
            errors.append(f"dataset {dataset_name}['features'] should be a dict")
        if not isinstance(dataset['targets'], dict):
            errors.append(f"dataset {dataset_name}['targets'] should be a dict")
        if not isinstance(dataset['covariates'], dict):
            errors.append(f"dataset {dataset_name}['covariates'] should be a dict")
        
        # check that the dataset keys have the expected values
        for feature in dataset['features']:
            if not isinstance(feature, str):
                errors.append(f"dataset {dataset_name} feature {feature} should be a list of strings")
            if '_' in feature:
                errors.append(f"dataset {dataset_name} feature {feature} should be a list of strings without underscores")
        for target in dataset['targets']:
            if not isinstance(target, str):
                errors.append(f"dataset {dataset_name} target {target} should be a list of strings")
            if '_' in target:
                errors.append(f"dataset {dataset_name} target {target} should be a list of strings without underscores")
        for covariate in dataset['covariates']:
            if not isinstance(covariate, str):
                errors.append(f"dataset {dataset_name} covariate {covariate} should be a list of strings")
            if '_' in covariate:
                errors.append(f"dataset {dataset_name} covariate {covariate}  should be a list of strings without underscores")
        
        # check that each path exists
        for feature in dataset['features']:
            if not os.path.exists(dataset['features'][feature]):
                errors.append(f"dataset {dataset_name} feature {feature} contains a feature file that does not exist: {feature}")
        for target in dataset['targets']:
            if not os.path.exists(dataset['targets'][target]):
                errors.append(f"dataset {dataset_name} target {target} contains a target file that does not exist: {target}")

    return errors

# read yaml config file
import yaml
# with open('config/config.yaml', 'r') as f:
with open('example/example_config.yaml', 'r') as f:
    config = yaml.load(f)
print(config)
errors = validate_root(config)+validate_experiments(config)+validate_custom_datasets(config)
for error in errors:
    print(error)    
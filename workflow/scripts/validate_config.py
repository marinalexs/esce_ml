import os

import yaml


def validate_details(config: dict, MODELS: dict, PREDEFINED_DATASETS: dict, RegressionModel: type) -> list:
    """
    Validate the details of the configuration file to ensure consistency and correctness.

    This function checks that each experiment defined in the configuration has valid datasets,
    features, targets, confound corrections, models, and corresponding hyperparameter grids.
    It also verifies the existence of custom dataset files.

    Args:
        config (dict): Parsed configuration dictionary.
        MODELS (dict): Dictionary of available models.
        PREDEFINED_DATASETS (dict): Dictionary of predefined datasets.
        RegressionModel (type): Class reference for regression models.
    
    Returns:
        list: List of error messages indicating validation failures.
    """
    
    errors = []
    
    # Get global settings
    global_grid = config.get("grid", "default")
    global_balanced = config.get("balanced", False)
    global_quantile_transform = config.get("quantile_transform", False)
    
    # Iterate over each experiment in the configuration
    for exp_name, exp in config["experiments"].items():
    
        # Check if the dataset exists in predefined or custom datasets
        if (
            exp["dataset"] not in PREDEFINED_DATASETS
            and exp["dataset"] not in config["custom_datasets"]
        ):
            errors.append(
                f"Experiment '{exp_name}': dataset '{exp['dataset']}' should be one of {list(PREDEFINED_DATASETS.keys())} or {list(config['custom_datasets'].keys())}."
            )
        else:
            # Retrieve the dataset details
            dataset = (
                PREDEFINED_DATASETS[exp["dataset"]]
                if exp["dataset"] in PREDEFINED_DATASETS
                else config["custom_datasets"][exp["dataset"]]
            )
            # Validate features
            for feature in exp["features"]:
                if feature not in dataset["features"]:
                    errors.append(
                        f"Experiment '{exp_name}': feature '{feature}' must exist in the predefined or custom datasets."
                    )
            # Validate confound corrections
            for confound_correction_cni in exp["confound_correction_cni"]:
                if confound_correction_cni != "none" and confound_correction_cni not in dataset["covariates"]:
                    errors.append(
                        f"Experiment '{exp_name}': confound_correction_cni '{confound_correction_cni}' must exist in the predefined or custom datasets."
                    )
            # Validate targets
            for target in exp["targets"]:
                if target not in dataset["targets"]:
                    errors.append(
                        f"Experiment '{exp_name}': target '{target}' must exist in the predefined or custom datasets."
                    )
    
        # Check if the model exists in the available models
        for model in exp["models"]:
            if model not in MODELS:
                errors.append(
                    f"Experiment '{exp_name}': model '{model}' is not defined in the available MODELS."
                )
    
        # Validate hyperparameter grids
        exp_grid = exp.get("grid", global_grid)
        if exp_grid not in config['grids']:
            available_grids = os.listdir('config/grids')
            errors.append(
                f"Experiment '{exp_name}': grid '{exp_grid}' should be one of {available_grids}."
            )
        else:
            grid = config['grids'][exp_grid]
            for model in exp["models"]:
                if model not in grid:
                    errors.append(
                        f"Experiment '{exp_name}': grid '{exp_grid}' must have hyperparameters defined for model '{model}'."
                    )
    
        # Ensure regression models are used appropriately with confound corrections
        if exp["confound_correction_method"] in ["correct-y", "correct-both"]:
            for model in exp["models"]:
                if not isinstance(MODELS[model], RegressionModel):
                    errors.append(
                        f"Experiment '{exp_name}': model '{model}' must be a regression model when confound_correction_method is '{exp['confound_correction_method']}'."
                    )
    
    # Validate the existence of custom dataset files
    for dataset_name, dataset in config["custom_datasets"].items():
        for feature, feature_path in dataset["features"].items():
            if not os.path.exists(feature_path):
                errors.append(
                    f"Dataset '{dataset_name}': feature file '{feature}' does not exist at path '{feature_path}'."
                )
        for target, target_path in dataset["targets"].items():
            if not os.path.exists(target_path):
                errors.append(
                    f"Dataset '{dataset_name}': target file '{target}' does not exist at path '{target_path}'."
                )
    
    return errors


if __name__ == "__main__":
    import importlib.util  
    
    # Dynamically import the 'fit_model' module
    spec = importlib.util.spec_from_file_location('fit_model', snakemake.input.fit_model)
    module_fit_model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_fit_model)
    
    # Dynamically import the 'prepare_data' module
    spec = importlib.util.spec_from_file_location('prepare_data', snakemake.input.prepare_data)
    module_prepare_data = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module_prepare_data)
    
    from fit_model import MODELS, RegressionModel
    from prepare_data import predefined_datasets as PREDEFINED_DATASETS
    
    # Validate the configuration details
    errors = validate_details(snakemake.params.config, MODELS, PREDEFINED_DATASETS, RegressionModel)
    
    # Print all encountered errors
    for error in errors:
        print(error)
    
    # Raise an error if any validation checks failed
    if len(errors) > 0:
        raise ValueError("Configuration file has errors.")
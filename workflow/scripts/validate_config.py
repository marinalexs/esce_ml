import os

import yaml


def validate_details(config, MODELS, PREDEFINED_DATASETS, RegressionModel):
    """Validate the details of the config file.

    Args:
        config: path to the config file
        MODELS: dictionary of models
        PREDEFINED_DATASETS: dictionary of predefined datasets
        RegressionModel: class of regression models (this is a hack to circumvent smakemake import issues)
    """

    errors = []

    # check that each experiment has the expected values
    for exp_name, exp in config["experiments"].items():

        # check that dataset exists
        if (
            exp["dataset"] not in PREDEFINED_DATASETS
            and exp["dataset"] not in config["custom_datasets"]
        ):
            errors.append(
                f"experiment {exp_name}['dataset'] should be a string from: {PREDEFINED_DATASETS.keys()} or {config['custom_datasets'].keys()}"
            )
        else:
            # check that all features, targets, covariates exist
            dataset = (
                PREDEFINED_DATASETS[exp["dataset"]]
                if exp["dataset"] in PREDEFINED_DATASETS
                else config["custom_datasets"][exp["dataset"]]
            )
            for feature in exp["features"]:
                if feature not in dataset["features"]:
                    errors.append(
                        f"experiment {exp_name} feature {feature} must exist in either predefined  or custom datasets"
                    )
            for confound_correction_cni in exp["confound_correction_cni"]:
                if confound_correction_cni != "none" and confound_correction_cni not in dataset["covariates"]:
                    errors.append(
                        f"experiment {exp_name} confound_correction_cni {confound_correction_cni} must exist in either predefined  or custom datasets"
                    )
            for target in exp["targets"]:
                if target not in dataset["targets"]:
                    errors.append(
                        f"experiment {exp_name} target {target} must exist in either predefined  or custom datasets"
                    )

        # check that model exists in MODELS
        for model in exp["models"]:
            if model not in MODELS:
                errors.append(
                    f"experiment {exp_name}['models'] should exist in: {MODELS}"
                )

        # check that grid has a corresponding yaml file in config/grids
        # and that all models have a key in the grid
        if  exp["grid"] not in config['grids']:
            errors.append(
                f"experiment {exp_name}['grid'] should be a string from: {os.listdir('config/grids')}"
            )
        else:
            grid = config['grids'][exp["grid"]]
            for model in exp["models"]:
                if model not in grid:
                    errors.append(
                        f"experiment {exp_name}['grid'] should have hyperparameters defined for model: {model}"
                    )

        # if target_cni is not none, then matching must be none and model must be regression
        if exp["confound_correction_method"] in ["correct-y", "correct-both"]:
            for model in exp["models"]:
                if not isinstance(MODELS[model], RegressionModel):
                    errors.append(
                        f"experiment {exp_name} model {model} must be regression model when {exp_name} targets_cni is not 'none'"
                    )

    # check that each path exists
    for dataset_name, dataset in config["custom_datasets"].items():
        for feature in dataset["features"]:
            if not os.path.exists(dataset["features"][feature]):
                errors.append(
                    f"dataset {dataset_name} feature {feature} contains a feature file that does not exist: {feature}"
                )
        for target in dataset["targets"]:
            if not os.path.exists(dataset["targets"][target]):
                errors.append(
                    f"dataset {dataset_name} target {target} contains a target file that does not exist: {target}"
                )

    return errors


if __name__ == "__main__":
    import importlib.util  

    spec = importlib.util.spec_from_file_location('fit_model', snakemake.input.fit_model)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    spec = importlib.util.spec_from_file_location('prepare_data', snakemake.input.prepare_data)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    from fit_model import MODELS, RegressionModel
    from prepare_data import predefined_datasets as PREDEFINED_DATASETS

    errors = validate_details(snakemake.params.config, MODELS, PREDEFINED_DATASETS, RegressionModel)

    for error in errors:
        print(error)

    if len(errors) > 0:
        raise ValueError("config file has errors")
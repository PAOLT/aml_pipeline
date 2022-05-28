# Azure ML Pipeline template

This repository is about setting up a reusable template to an Azure ML pipeline for data preparation and training.

Useful references:

[How to create machine learning pipelines](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines)

[Tutorial: pipeline python sdk](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk)

## Structure of the repository

The repository has three main folders, as explained below.

### AML Pipeline code

- `pipeline/pipe_scripts` holds the code for a two stages pipeline (data prep and train).
- `pipeline/args.yml` is a configuration file to provision the pipeline
- `pipeline/pipe_definition_script.py` provisions and run the pipeline

Note that the actual business logic for data preparation is separated from AML scripts, for easier testing.

### A toy package

`some_model_package` is a toy model class that is used for training by the pipeline.
Note that this package will be installed in the remote AML environment.

### Unit Tests

`tests` holds the unit tests. Note that the code for data preparation and model training are
separated from the AML code.

## Set up VSCode

Create a Python environment ([cfr. this](https://docs.microsoft.com/en-us/python/api/overview/azure/ml/install?view=azure-ml-py))

```bash
conda create -n <name> python=3.7
pip install -r requirements.txt
```

Download `config.json` from the Azure ML workspace web site and store it into the `.azureml` folder.

Add an `.env` file with paths where to search for packages. In this case we have a `some_model_package` module in the project's root folder, hence we added `PYTHONPATH = C:\Users\paolt\Documents\Dev\aml_pipeline` to `.env` so that we can use, for example, `from some_model_package.some_model import SomeModel` ([cfr. this](https://code.visualstudio.com/docs/python/environments#_use-of-the-pythonpath-variable)).

Add a debug configuration for the current Python file. Add `"cwd": "${workspaceFolder}"` to it to set the current working directory to the root of the project.

[Build a Python wheel](https://medium.com/swlh/beginners-guide-to-create-python-wheel-7d45f8350a94)
for the package `somemodel` by running `python setup.py bdist_wheel`,
and install it by running `pip install somemodel-3.0-py3-none-any.whl`
(where `somemodel-3.0-py3-none-any.whl` is the wheel generated by the previous command under `dist`).

## Set up AML

Create an Azure ML *Workspace* and save `config.json` to `.azureml` in the root folder of the project.

Create a *Datastore* and name it `training_datastore` (we haven't parametrized this).

Copy `tests/data/iris_dataset.csv` to the Datastore

Create a *Compute* instance (a Data Science VM is fine).

Create a base *Environment* and register it to the Workspace.
You can include in the environment the Python packages enlisted in `requirements.txt`,
or use a [curated environment](https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments)
and add the missing packages afterwards.

## How to run

To run the pipeline, fill in a `pipeline/args.yml` file ( similar to `pipeline/args_template.yml`):

`tenant`: optionally set to the tenant where the relevant Azure subscription resides.

`dataset_path`: the relativ path to `iris_dataset.csv` in the Datastore.

`experiment_name`: set this to any experiment name.

`compute`: set this to the name of the AML compute resource you have created.

`base_environment`: set this to the name of the AML environment resource you have created.

`conda_packages`, `pip_packages` and `conda_channels`: add the packages that have not been included in the above AML environment.

`local_package_wheel`: set this to the path of the wheel file (`.whl`)

Finally run `pipeline/pipe_definition_script.py`. To monitor the execution follow the workspace link
to the run, returned by the script.

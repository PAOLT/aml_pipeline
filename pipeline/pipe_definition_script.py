from pathlib import Path

import azureml.core
import yaml
from azureml.core import Datastore, Environment, Experiment, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication
from azureml.core.runconfig import RunConfiguration
from azureml.data import OutputFileDatasetConfig
from azureml.pipeline.core import Pipeline, PipelineParameter
from azureml.pipeline.steps import PythonScriptStep

# check core SDK version number
print("\nAzure ML SDK Version: ", azureml.core.VERSION)

# # Instantiate a model
# from some_model_package.some_model import SomeModel
# some_model = SomeModel(name="my-model")
# print(some_model)

root_path = Path.cwd()
print (f"\nWorking directory: {Path.cwd()}")

# Parse arguments
with open(str(Path(__file__).parent.absolute() / 'args.yml')) as f:
    args = yaml.safe_load(f)

data_prep_folder = root_path / "pipeline" / "pipe_scripts" / args['data_prep_folder']
data_prep_file = args['data_prep_script']

train_folder = root_path / "pipeline" / "pipe_scripts" / args['train_folder']
train_file = args['train_script']

compute_target_name = args['compute']
experiment_name = args['experiment_name']
base_environment_name = args["base_environment"]
conda_packages = args['conda_packages']
pip_packages = args['pip_packages']
conda_channels = args['conda_channels']
local_package_wheel = args['local_package_wheel']
tenant_id = args["tenant"]
dataset_path = args["dataset_path"] # this one should be passed via API

# Get AML workspace reference
interactive_auth = InteractiveLoginAuthentication(
    tenant_id=tenant_id) if tenant_id else None
ws = Workspace.from_config(auth=interactive_auth)
print(f"\nWorkspace information: {ws}")

# Set an experiment
experiment = Experiment(ws, experiment_name)
print(f"\nExperiment information: {experiment}")

# Instantiate a run config
aml_run_config = RunConfiguration()

# Set compute target
if compute_target_name == 'local':
    compute_target = 'local'
else:
    compute_target = ws.compute_targets[compute_target_name]
aml_run_config.target = compute_target

# Set the environment
run_environment = Environment.get(
    ws, name=base_environment_name).clone("env-pipeline")  

if conda_packages:
    conda_packages = [dep.lower().strip()
                        for dep in conda_packages.split(' ')]
    for d in conda_packages:
        run_environment.python.conda_dependencies.add_conda_package(d)

if pip_packages:
    pip_packages = [dep.lower().strip()
                    for dep in pip_packages.split(' ')]
    for d in pip_packages:
        run_environment.python.conda_dependencies.add_pip_package(d)

if conda_channels:
    conda_channels = [ch.lower().strip()
                        for ch in conda_channels.split(' ')]
    for c in conda_channels:
        run_environment.python.conda_dependencies.add_channel(c)

if local_package_wheel:
    whl_path = root_path / local_package_wheel
    whl_url = Environment.add_private_pip_wheel(
        workspace=ws, file_path=whl_path, exist_ok=True)
    run_environment.python.conda_dependencies.add_pip_package(whl_url)

print(run_environment)
aml_run_config.environment = run_environment


# # Get the default data store for the workspace
def_blob_store = Datastore(ws, "training_datastore")
print(f"Default data-store: {def_blob_store}")

# # Define a tabular dataset
# ds_input = Dataset.Tabular.from_delimited_files(path = [(def_blob_store, dataset_path)])
# print(f"\nData sample: {ds_input.take(5).to_pandas_dataframe()}")




# Define the pipeline

# Prep-data input
input_dataset_path_param = PipelineParameter(name="input_dataset_path", default_value=dataset_path)
  
# Prep-data stage output / Training stage input
prepared_dataframe = OutputFileDatasetConfig(destination = (def_blob_store, 'prepared_dataframes/{run-id}.pkl'))

# training output
trained_model = OutputFileDatasetConfig(name = "trained_model", destination = (def_blob_store, 'trained_models/{run-id}.pkl'))

# Data-prep stage
data_prep_step = PythonScriptStep(
    source_directory = data_prep_folder,
    script_name = data_prep_file,
    arguments=["--input-dataset-path", input_dataset_path_param, "--output-dataframe-path", prepared_dataframe],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

# Train stage
train_step = PythonScriptStep(
    source_directory = train_folder,
    script_name = train_file,
    arguments=["--training-dataframe", prepared_dataframe.as_input(), "--trained-model", trained_model],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

# Build and publish the pipeline
pipeline = Pipeline(workspace=ws, steps=[data_prep_step, train_step])
run = experiment.submit(pipeline, pipeline_parameters={"input_dataset_path": dataset_path})

# published_pipeline = pipeline.publish(name="Train_Pipeline",
#                                          description="Train Pipeline",
#                                          version="1.0",
#                                          continue_on_step_failure=False)

from pathlib import Path

import azureml.core
import yaml
from azureml.core import Experiment, Workspace
from azureml.core.authentication import InteractiveLoginAuthentication

from construct_pipeline import construct_pipeline

# check core SDK version number
print("\nAzure ML SDK Version: ", azureml.core.VERSION)

# Parse arguments
with open(str(Path(__file__).parent.absolute() / 'args.yml')) as f:
    args = yaml.safe_load(f)

# Get AML workspace reference
tenant_id = args["tenant"]
dataset_path = args["dataset_path"] # this one should be passed via API
experiment_name = args['experiment_name']

interactive_auth = InteractiveLoginAuthentication(
    tenant_id=tenant_id) if tenant_id else None
ws = Workspace.from_config(auth=interactive_auth)
print(f"\nWorkspace information: {ws}")

# Set an experiment
experiment = Experiment(ws, experiment_name)
print(f"\nExperiment information: {experiment}")

pipeline = construct_pipeline(ws, args)

run = experiment.submit(pipeline, pipeline_parameters={"input_dataset_path": dataset_path})

# published_pipeline = pipeline.publish(name="Train_Pipeline",
#                                          description="Train Pipeline",
#                                          version="1.0",
#                                          continue_on_step_failure=False)

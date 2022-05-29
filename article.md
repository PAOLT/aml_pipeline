# Title

Since recently I'm working to a project for operationalizing a ML classification model, where we have to run inference over hundreds of thousands of records every week. We have been looking at Azure ML (AML) pipelines, and this post is about sharing our toy prototype.
I think this can be useful as a template for those willing to use AML pipelines in a similar context.

To make things simple let's consider a two stages pipeline for data preparation and training. Other stages (such as evaluation and testing) can be added seamlessy. Data preparation is about reading one or more CSV files from a cloud storage, and processing them. In our simple pipeline we are using the well known *irsi dataset*. Training uses a toy model exposed through a Python package external to the pipeline project (i.e., it is a Python wheel), and produces a pickle file with the trained mode over a cloud storage. The pipeline will be published to AML and exposed through a REST API. The path to the input CSV file can be provided to the API as a parameter. So, to re-train the model is sufficien to invoke the API by passing the CSV file path, and get the pickle file with the trained model in a standard place. As said this is just a template over sample data and a toy model, but it can be easly adapted to any batch ML workflow. For example, inference would work the same way, with the input being the CSV file(s) with input data and the trained model, and output would be another CSV file with scored data.

All the artifacts are publicly available in [this GitHub repository](https://github.com/PAOLT/aml_pipeline). The accompaining `README file` provides practical information to create AML artifacts, configure VSCode, and run the pipeline. Find [here](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-visual-studio-code) how to debug the pipeline. We also added unit tests and an external package wheel.

## Organization of the repository

The project structure is fairly simple.

The `pipeline` folder holds all the AML pipeline scripts. Specifically, one script to construct and run the pipeline (`pipe_definition_script.py`, described below), and one script per pipeline stage (`pipeline/pipe_scripts`).

The training stage's script references an external package (`somemodel`):

```Python
from somemodel.models import SomeModel
```

The `some_model_package` folder, in the root of the project, holds the source code for `somemodel`, and the Python *wheel*, that is required to deploy the package to the AML runtime through an *Environment* object.

The data processing stage has also some logic in the `data_prep()` function, that is not covered by `somemodel` and has been put in its own Python module.

```Python
def data_prep(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame
```

 By keeping `data_prep()` separated from the AML specific code in the stage's script (i.e., managing inputs and outputs thorugh AML objects) it can be easily debugged and unit tested, and eventually included in `somemodel` at a later stage.

## Script to construct and run the pipeline

`pipe_definition_script.py` contains all the code necessary to construct the pipeline and run it, based on parameters stored in a `yaml` file. So, to run the pipeline it is necessary to compile parameters in the `yaml` file and run the Python script.

### Infrastructure to run the Pipeline

First the script performs [interactive authentication](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication#use-interactive-authentication) to AML (`interactive_auth`) and creates a `Workspace` object from a configuration file stored locally. In addition, it creates an `Experiment` object to log runs of the pipeline, and a `RunConfiguration` object to store the runtime configuration (i.e., a [compute](https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-v2?tabs=sdk#compute) and an [environment](https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-v2?tabs=sdk#environment)).

```Python
# tenant_id and experiment_name read from yaml
interactive_auth = InteractiveLoginAuthentication(
    tenant_id=tenant_id) if tenant_id else None
ws = Workspace.from_config(auth=interactive_auth)
experiment = Experiment(ws, experiment_name)
aml_run_config = RunConfiguration()
```

To run the pipeline, AML uses the `RunConfiguration` object to provisions a runtime, and the `Experiment` object as a scope for the run. Thus, the scripts stores a compute target to the `RunConfiguration` object. The compute target is  `local` to run the pipeline on the local computer, or a proper compute (e.g., a VM) registerd in the AML Workspace (compute targets should be provisioned through a proper CI/CD pipeline).

```Python
# compute_target_name read from yaml
if compute_target_name == 'local':
    compute_target = 'local'
else:
    compute_target = ws.compute_targets[compute_target_name]
aml_run_config.target = compute_target
```

In addition an `Environment` object is also associated to the `RunConfiguration`. A base environment could already be available in the AML Workspace: beside some [curated environments](https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments) already available in AML, they can be constructed and registered for later reuse.

```Python
# base_environment_name read from yaml
run_environment = Environment.get(
    ws, name=base_environment_name).clone("env-pipeline")  
```

Whatever the environment is get from, it should be cloned to eventually add additional `pip` or `conda` packages.

```Python
# conda_packages read from yaml
if conda_packages:
    conda_packages = [dep.lower().strip()
                        for dep in conda_packages.split(' ')]
    for d in conda_packages:
        run_environment.python.conda_dependencies.add_conda_package(d)

# pip_packages read from yaml
if pip_packages:
    pip_packages = [dep.lower().strip()
                    for dep in pip_packages.split(' ')]
    for d in pip_packages:
        run_environment.python.conda_dependencies.add_pip_package(d)

# conda_channels read from yaml
if conda_channels:
    conda_channels = [ch.lower().strip()
                        for ch in conda_channels.split(' ')]
    for c in conda_channels:
        run_environment.python.conda_dependencies.add_channel(c)
```

Finally, the `somemodel` package is also added to the environment, by pointing to the wheel `.whl` file and adding it as a `pip` dependency.

```Python
# local_package_wheel read from yaml
if local_package_wheel:
    whl_path = root_path / local_package_wheel
    whl_url = Environment.add_private_pip_wheel(
        workspace=ws, file_path=whl_path, exist_ok=True)
    run_environment.python.conda_dependencies.add_pip_package(whl_url)
```

### Management of parameters

The script then creates AML objects to represent parameters. We are reading input data (a CSV file) and writing outputs (a pickle file) as BLOBs in an Azure Storage Account, that is registered in the AML Workspace as a [Datastore](https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-v2?tabs=sdk#datastore).

```Python
# datastore_name read from yaml
def_blob_store = Datastore(ws, datastore_name)
```

Input data to train the model (i.e., the CSV file) will be different for every run of the pipeline. To this aim, we need to create a `PipelineParameter` object such that, when the pipeline will be published, the AML infrastructure will provision an end-point with a parameter. `dataset_path` is a relative path to the input data file in the AML datastore.

```Python
# dataset_path read from yaml
input_dataset_path_param = PipelineParameter(name="input_dataset_path", default_value=dataset_path)
```

The pipeline has some intermediate data, that is, the first stage outputs a file with prepared data, that will be inputed to the second stage for actual training. Note that intermediate data will be stored to the AML datastore `def_blob_store` under the path `prepared_dataframes/{run-id}.pkl` (where `run-id` will be instantiated by the AML run time with the pipeline run id). Very similarly, the second stage outputs a trained model as a file in the AML datastore.

```Python
prepared_dataframe = OutputFileDatasetConfig(destination = (def_blob_store, 'prepared_dataframes/{run-id}.pkl'))
trained_model = OutputFileDatasetConfig(name = "trained_model", destination = (def_blob_store, 'trained_models/{run-id}.pkl'))
```

In the next section we will see how these parameters are passed to the pipeline stages. For the moment, it is important to note that parameters are just AML objects that represent BLOBs by abstracting their implementation details. This is very convenient, because we do not have to explicitly deal with BLOBs, i.e., compiling their URIs and connection strings (instead, we refer to a datastore registered in the workspace), authenticating (instead, we authenticated to the workspace at the beginning of the script), performing explicit read/write operations (instead, parameters objects present themselves as simple files). For example, the script for training uses `prepared_dataframe` and `trained_model` as follow:

```Python
parser = argparse.ArgumentParser()
parser.add_argument('--training-dataframe', type=str, dest='training_dataframe_path', help='path to the input dataframe for training')
parser.add_argument('--trained-model', type=str, dest='trained_model_path', help='path to the trained model')
args = parser.parse_args()

training_dataframe_path = args.training_dataframe_path
trained_model_path = args.trained_model_path

input_pickle_path = Path(training_dataframe_path) / 'prepared_data.pkl'
output_pickle_path = Path(trained_model_path) / 'trained_model.pkl'

with open(input_pickle_path, "rb") as f:
    (df_data, labels) = pickle.load(f) 

model = SomeModel()
trained_model = model.train(df_data, labels)

with open(output_pickle_path, "wb") as f:
    pickle.dump(model.get_model(), f)
```

Note that parameters are acquired as command line arguments (this is not the only way, as they could be captured from the run session, though this approach seemed to us more generalizable).

For example, AML injects `training_dataframe_path` from the parameter `trained_model`, and the script uses it as a simple path to construct `input_pickle_path`. Data for training is hence read from there: `with open(input_pickle_path, "rb") as f:`.

### Creating the pipeline

The pipeline has two stages as `PythonScriptStep` objects. Note that they receive the path to the relavant scripts (`source_directory` and `script_name`) and, for the rest, all the objects constructed above (i.e., the list of parameters to pass as arguments (`arguments`), the `RunConfig` object (`runconfig`) and the compute target (`compute_target`).

```Python
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

# Pipeline
pipeline = Pipeline(workspace=ws, steps=[data_prep_step, train_step])
```

## Run and publish

Running the pipeline is easy (note that the varying parameter is passed):

```Python
run = experiment.submit(pipeline, pipeline_parameters={"input_dataset_path": 'some/relative/path'})
```

The pipeline is run in the scope of an AML Experiment, where it is logged.

Having to re-run the script every time we want to re-run the pipeline would be error prone and hard to integrate. Hence, the pipeline could be registered to the AML workspace and a REST endpoint automatically created. This way, to run the pipeline it's enough to call a REST API by passing `input_dataset_path` in the body of the call.

```Python
published_pipeline = pipeline.publish(name="Train_Pipeline",
                                         description="Train Pipeline",
                                         version="1.0",
                                         continue_on_step_failure=False)
```

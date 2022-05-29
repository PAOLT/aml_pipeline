# Title

Since recently I'm working to a project for operationalizing a ML classification model, where we have to run batch inference every week. We have been looking at Azure ML (AML) pipelines, and this post is about sharing our toy prototype. I hope this can be useful as a template for those willing to use AML pipelines in a context similar to the ours:

- **Training**: eventually we get some input data in the forms of one or more CSV files stored as BLOBs in an Azure Storage Account, we train a ML model, and we store the trained model as a serialized object in another BLOB, in the same Storage Account.
- **Inference**: once per week we get some input data in the forms of one or more CSV files stored as BLOBs in an Azure Storage Account, we deserialize the last trained model from the relevant BLOB, make predictions and store them as a BLOB, in the same Storage Account.

AML Pipelines allows to model the workflows above, publish them as a REST endpoints, run and monitor them over cloud resources.
To prototype a toy AML Pipeline, our use case could be generalized as follow:

- We get some input data as a BLOB in a Storage Account (a single CSV file with the popular *irsi dataset*).
- We do some data pre-processing
- We do model training over the pre-processed data
- We store the serialized trained model as a BLOB in a Storage Account.

We considered the following aspects for generalization:

- The input BLOB will vary for every run of the pipeline, hence the REST end-point should accept it as a parameter. The trained model also varies from run to run, however the target BLOB can be either overwritten, or named after the run id.
- The model comes with an external Python package (this is our actual case, as our classification model is a custom class).

All the artifacts of the toy pipeline are publicly available in a GitHub [repository](https://github.com/PAOLT/aml_pipeline). The accompaining `README file` provides practical information to create AML artifacts, configure VSCode, and run the pipeline. The process to debug AML pipelines are described in a dedicated [article](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-debug-visual-studio-code).

## Organization of the repository

The repository holds, among the others, one folder (`pipeline`) with all the AML pipeline related artifacts. These are Python scripts:

- `pipe_definition_script.py` constructs the pipeline and run it (described below in details)
- `pipe_register_script.py` publish the pipeline to AML and run it via the REST endpoint.
- `pipeline/pipe_scripts` holds the scripts and helpers that define the pipeline's stages, that is, `pipeline/pipe_scripts/data_prep_stage` for data preparation and `pipeline/pipe_scripts/train_stage` for training.

The training stage's script references an external package (`somemodel`):

```Python
from somemodel.models import SomeModel
```

Note that the pipeline will be run over AML compute resources, that knows nothing about `somemodel`, thus, it will have to be *pip-installed* in the target AML compute resources (AML provides a way to automate this, as we will see).

The data processing stage has also some business logic in the `data_prep()` function, that is not covered by `somemodel` and has been put in its own Python module within the stage's folder.

```Python
def data_prep(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame
```

By keeping `data_prep()` separated from the AML specific code in the stage's script (i.e., managing inputs and outputs thorugh AML objects) it can be easily debugged and unit tested, and eventually included in `somemodel` at a later stage.

Note that `data_prep()` will be deployed to the AML compute resources together with the stage's script, thus it doesn't need to be *pip-installed*.

## Constructing the pipeline

`pipe_definition_script.py` contains all the code necessary to construct the pipeline and run it, based on parameters stored in a `yaml` file.

### Infrastructure to run the Pipeline

First the script performs [interactive authentication](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-setup-authentication#use-interactive-authentication) to AML (`interactive_auth`) and creates a `Workspace` object from a configuration file stored locally.

```Python
# tenant_id and experiment_name read from yaml
interactive_auth = InteractiveLoginAuthentication(
    tenant_id=tenant_id) if tenant_id else None
ws = Workspace.from_config(auth=interactive_auth)
```

 An `Experiment` object to scope runs of the pipeline is also created:

```Python
experiment = Experiment(ws, experiment_name)
```

In addition a `RunConfiguration` object is created to store the runtime configuration for the pipeline (i.e., a [compute](https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-v2?tabs=sdk#compute) and an [environment](https://docs.microsoft.com/en-us/azure/machine-learning/concept-azure-machine-learning-v2?tabs=sdk#environment)). Specifically, a `RunConfiguration` object is used by AML to provision a runtime as a container deployed to the compute resources.

```Python
aml_run_config = RunConfiguration()
```

The pipeline is run over cloud resources, i.e., AML compute. This can be `local` to run the pipeline on the local computer for debugging, or a proper compute (e.g., a VM or a cluster). Note that the latter would be already registerd in the AML Workspace as compute targets should be provisioned through a proper CI/CD pipeline.

```Python
# compute_target_name read from yaml
if compute_target_name == 'local':
    compute_target = 'local'
else:
    compute_target = ws.compute_targets[compute_target_name]
aml_run_config.target = compute_target
```

In addition an `Environment` object is also associated to the `RunConfiguration`. A base environment would already be available in the AML Workspace: beside some [curated environments](https://docs.microsoft.com/en-us/azure/machine-learning/resource-curated-environments) already available in AML, they can be constructed and registered for later reuse.

```Python
# base_environment_name read from yaml
run_environment = Environment.get(
    ws, name=base_environment_name).clone("env-pipeline")  
```

Whatever the environment is curated or not, it should be cloned to eventually add additional `pip` or `conda` packages.

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

Finally, the `somemodel` package is also added to the environment, by pointing to the wheel `.whl` file and adding it as a `pip` dependency. This will *pip-install* `somemodel` to the target environment in the compute's container.

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

The pipeline has some intermediate data, that is, the first stage outputs a file with prepared data, that will be inputed to the second stage for actual training. Intermediate data will be stored to the AML datastore `def_blob_store` under the path `prepared_dataframes/{run-id}.pkl` (where `run-id` will be instantiated by the AML run time with the pipeline run id). Very similarly, the second stage outputs a trained model as a file in the AML datastore.

```Python
prepared_dataframe = OutputFileDatasetConfig(destination = (def_blob_store, 'prepared_dataframes/{run-id}.pkl'))
trained_model = OutputFileDatasetConfig(name = "trained_model", destination = (def_blob_store, 'trained_models/{run-id}.pkl'))
```

In the next section we will see how these parameters are passed to the pipeline stages. For the moment, it is important to note that parameters are just AML objects that represent BLOBs by abstracting their implementation details. This is very convenient, because we do not have to explicitly deal with BLOBs, i.e., compiling their URIs and connection strings (instead, we refer to a datastore registered in the workspace), authenticating (instead, we authenticated to the AML workspace at the beginning of the script), performing explicit read/write operations (instead, parameters objects present themselves as simple files).

For example, let's consider the `trained_model` parameters in the training stage's script. First it is acquired as a command line argument (this is not the only way, as they could be captured from the run session, though this approach seemed to us more generalizable):

```Python
parser = argparse.ArgumentParser()
parser.add_argument('--trained-model', type=str, dest='trained_model_path', help='path to the trained model')
args = parser.parse_args()
trained_model_path = args.trained_model_path
```

Then it is treated as it was a file path:

```Python
output_pickle_path = Path(trained_model_path) / 'trained_model.pkl'

# some work happens here that produces trained_model_object

with open(output_pickle_path, "wb") as f:
    pickle.dump(trained_model_object, f)
```

### Creating the pipeline

The pipeline has two stages represented by `PythonScriptStep` objects. Note that they receive the path to the relavant scripts (`source_directory` and `script_name`) and, for the rest, all the objects constructed above are passed to the constructor.

```Python
# Data-prep stage
data_prep_step = PythonScriptStep(
    source_directory = 'data/prep/folder',
    script_name = 'data/prep/file.py',
    arguments=["--input-dataset-path", input_dataset_path_param, "--output-dataframe-path", prepared_dataframe],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

# Train stage
train_step = PythonScriptStep(
    source_directory = 'train/folder',
    script_name = 'train/file.py',
    arguments=["--training-dataframe", prepared_dataframe.as_input(), "--trained-model", trained_model],
    compute_target=compute_target,
    runconfig=aml_run_config,
    allow_reuse=True
)

# Pipeline
pipeline = Pipeline(workspace=ws, steps=[data_prep_step, train_step])
```

### Run

The pipeline is run in the scope of an AML Experiment, where it is logged. Note that the varying parameter is passed.

```Python
run = experiment.submit(pipeline, pipeline_parameters={"input_dataset_path": 'some/relative/path'})
```

`experiment.submit()` returns a URL to the AML Portal, where the run is logged within `experiment`.

## Publish the pipeline

Having to re-run the script every time we want to re-run the pipeline would be error prone and hard to integrate. Hence, the pipeline can be registered to the AML workspace and a REST endpoint automatically created. This way, to run the pipeline it's enough to call a REST API by passing `input_dataset_path` in the body of the call.

```Python
published_pipeline = pipeline.publish(name="Train_Pipeline",
                                         description="Train Pipeline",
                                         version="1.0",
                                         continue_on_step_failure=False)
```

## Resources

Some usfeul resources would be the following:

- [Tutorial](https://docs.microsoft.com/en-us/azure/machine-learning/tutorial-pipeline-python-sdk#create-and-run-the-pipeline)
- [AML pipelines documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-create-machine-learning-pipelines)
- [AML SDK for Python](https://docs.microsoft.com/en-us/python/api/?view=azure-ml-py)

import argparse
import pickle
from pathlib import Path

import pandas as pd
from azureml.core import Dataset, Datastore, Run, Workspace

from data_preprocessing import data_prep


def get_workspace_from_run():
    run = Run.get_context()
    ws = run.experiment.workspace
    return ws

def get_data(ws: Workspace, dataset_path: str) -> pd.DataFrame:
    '''
    This is code used to retrieve data from a workspace, by means of a path
    '''

    def_blob_store = Datastore(ws, "training_datastore")
    ds_input = Dataset.Tabular.from_delimited_files(path = [(def_blob_store, dataset_path)])

    return ds_input.to_pandas_dataframe()

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dataset-path', type=str, dest='input_path', help='path to the input dataset')
    parser.add_argument('--output-dataframe-path', type=str, dest='output_path', help='path to the preprocessed dataset')
    args = parser.parse_args()
    input_path = args.input_path
    output_path = args.output_path

    ws = get_workspace_from_run()
    df = get_data(ws, input_path)

    data, labels = data_prep(df)

    print(data)
    print(labels)

    output_pickle_path = Path(output_path) / 'prepared_data.pkl'
    print(f"Writing data and labels to {output_pickle_path}")

    with open(output_pickle_path, "wb") as f:
        pickle.dump((data, labels), f)

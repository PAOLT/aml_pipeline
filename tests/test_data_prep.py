import pandas as pd
from pipeline.pipe_scripts.data_prep_stage.data_preprocessing import data_prep


def test_data_prep():
    iris = pd.read_csv("./tests/data/iris_dataset.csv")
    data, targets = data_prep(iris)
    assert len(data) == len(targets)
    labels = [0, 1, 2]
    assert all([t in labels for t in targets.unique()])

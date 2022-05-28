import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from some_model_package.somemodel.models import SomeModel


def test_train_test_model():
    data = load_iris()
    X = pd.DataFrame(data.data)
    y = pd.DataFrame(data.target)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
    model = SomeModel()        
    model.train(X_train, y_train)
    assert (model.get_model() != None)
    model.score(X_test, y_test)
    assert (model.get_accuracy() >= 0.0)



import pandas as pd

def data_prep(df: pd.DataFrame, target_col: str = 'target') -> pd.DataFrame:
    '''
    This is the code that should be tested. Unit tests will use a test df, without going through AML.
    '''
    # Pre-processing goes here

    data_columns = [col for col in df.columns if col != target_col]
    data = df[data_columns]

    d = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
    targets = df[target_col].apply(lambda r: d[r])
    
    return data, targets

# importing necessary libraries
import argparse
import pickle
from pathlib import Path

# import pandas as pd
# from sklearn.model_selection import train_test_split
from somemodel.models import SomeModel


# def train_model(data: pd.DataFrame):
    
#     X = data.data
#     y = data.target
    
#     # dividing X, y into train and test data
#     X_train, X_test, y_train, y_test = train_test_split(X, y, random_state = 0)
#     model = SomeModel()
#     model.train(X_train, y_train)
#     model.score(X_test, y_test)
#     print("\n*******************")
#     print("Model's results")
#     print(model.get_accuracy())
#     print("*******************\n")
#     return model.get_model()

if __name__ == "__main__":
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

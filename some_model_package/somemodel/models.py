from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier


class SomeModel:

    def __init__(self, name: str = "model x"):
        self.name = name
        self.dtree_model = None
        self.predictions = None
        self.y_test = None
        self.base_model = DecisionTreeClassifier(max_depth = 2)

    def train(self, X_train, y_train):
        self.dtree_model = self.base_model.fit(X_train, y_train)

    def score(self, X_test, y_test):
        if self.dtree_model:
            self.predictions = self.dtree_model.predict(X_test)
            self.y_test = y_test

    def get_accuracy(self):
        if self.predictions is not None and self.y_test is not None:
            return accuracy_score(self.y_test, self.predictions)
        else:
            return -1
    
    def get_model(self):
        return self.dtree_model

    def __str__(self):
        return f"This is {self.name}!"
    
    def __repr__(self):
        return self



 

# model_dispatcher.py

from sklearn import ensemble
from sklearn import tree

# creating a dictionary which contains all our models
# the keys are the names of the model and the values are the model themselves
models={
    "decision_tree_gini": tree.DecisionTreeClassifier(criterion="gini"),
    "decision_tree_entropy":tree.DecisionTreeClassifier(criterion="entropy"),
    "random_forest":ensemble.RandomForestClassifier()
}

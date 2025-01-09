import os
import pandas as pd
import rampwf as rw
from sklearn.model_selection import StratifiedGroupKFold

problem_title = "CASYS"
_target_column_name = "System"
_ignore_columns = ["#SampleID"]
_prediction_label_names = ["SD1", "SD2", "TS1", "TS2"]
# A type (class) which will be used to create wrapper objects for y_pred
Predictions = rw.prediction_types.make_multiclass(label_names=_prediction_label_names)
# An object implementing the workflow
workflow = rw.workflows.Classifier()

score_types = [
    rw.score_types.Accuracy(name="acc"),
]


def get_cv(X, y):
    cv = StratifiedGroupKFold(n_splits=5)
    groups = X[:, 0]
    return cv.split(X, y, groups=groups)


def _read_data(path, f_name):
    # We return NumPy array cause RAMP's cv method return integers index
    # and X[integer_index] would fail if X is a pd.DataFrame.
    data = pd.read_csv(os.path.join(path, "data", f_name))
    y_array = data[_target_column_name].to_numpy()
    X_df = data.drop(columns=[_target_column_name] + _ignore_columns)
    X_array = X_df.to_numpy()
    return X_array, y_array


def get_train_data(path="."):
    f_name = "train/train.csv"
    return _read_data(path, f_name)


def get_test_data(path="."):
    f_name = "test/test.csv"
    return _read_data(path, f_name)

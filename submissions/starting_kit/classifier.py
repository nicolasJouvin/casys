import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

_REMOVE_COL_IDX = [0]


def _preprocess_X(X):
    """
    This is a very basic preprocessor that only drops the "Parcelle" columns. A more refined pre-processing should be tried.
    """

    # Remove the `"Parcelle"` column from features (cannot be used)
    X_preprocessed = np.delete(X, _REMOVE_COL_IDX, axis=1)

    # Implement your own pre-preprocessing
    # ...

    return X_preprocessed


class Classifier(object):
    def __init__(self):
        # Use scikit-learn's pipeline
        custom_preprocessing = FunctionTransformer(
            func=_preprocess_X, inverse_func=None
        )
        self.pipe = Pipeline(
            [
                ("Custom preprocessing", custom_preprocessing),
                ("Scaler", StandardScaler(with_mean=True, with_std=True)),
                ("PCA with 50 components", PCA(n_components=20)),
                (
                    "Random Forest Classifier",
                    RandomForestClassifier(
                        max_depth=6, n_estimators=300, max_features=3
                    ),
                ),
            ]
        )

    def fit(self, X, y):
        self.pipe.fit(X, y)

        pass

    def predict_proba(self, X):
        # here we use RandomForest.predict_proba()
        return self.pipe.predict_proba(X)

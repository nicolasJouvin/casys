import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler, OneHotEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


_REMOVE_COL_IDX = [0]


def _get_otu(X):
    # --- Remove the `"Parcelle"` column from features (cannot be used)
    X_otu = np.delete(X, _REMOVE_COL_IDX, axis=1).astype(np.float64)
    return X_otu


def _filter_ind(X_otu):

    X_otu = _get_otu(X_otu)

    # remove least variables OTU
    otu_sd = X_otu.std(axis=0)

    # threshold on OTU sd
    sd_threshold = 1
    return np.argwhere(otu_sd > sd_threshold).squeeze()


def _preprocess_X(X_otu, keep_otu_ind):
    """
    This is a very basic preprocessor that only drops the "Parcelle" columns. A more refined pre-processing should be tried.
    """

    # --- Implement your own pre-preprocessing
    X_preprocessed = X_otu[:, keep_otu_ind]
    X_preprocessed = X_preprocessed / X_preprocessed.sum(axis=1, keepdims=True)
    return X_preprocessed


class Classifier(object):
    def __init__(self):
        self.label_encoder = OneHotEncoder(sparse_output=False).fit(
            np.array(["SD1", "SD2", "TS1", "TS2"]).reshape(-1, 1)
        )
        self.n_classes = 4

        # Use scikit-learn's pipeline

        self.pipe = Pipeline(
            [
                ("Scaler", StandardScaler(with_mean=True, with_std=True)),
                ("PCA with 50 components", PCA(n_components=50)),
                # (
                #     "Random Forest Classifier",
                #     RandomForestClassifier(
                #         max_depth=10, n_estimators=300, max_features=8
                #     ),
                # ),
                ("SVC", SVC()),
            ]
        )

    def fit(self, X, y):
        X_otu = _get_otu(X)
        self.keep_otu_ind_ = _filter_ind(X_otu)  # store this for .predict() step
        X_preprocessed = _preprocess_X(X_otu, self.keep_otu_ind_)
        self.pipe.fit(X_preprocessed, y)

        pass

    def predict_proba(self, X):
        X_otu = _get_otu(X)
        X_preprocessed = _preprocess_X(X_otu, self.keep_otu_ind_)
        # here we use RandomForest.predict_proba()
        # return self.pipe.predict_proba(X_preprocessed)

        # Or hard labels for SVC

        labels = self.pipe.predict(X_preprocessed)
        proba = self.label_encoder.transform(labels.reshape(-1, 1))
        return proba

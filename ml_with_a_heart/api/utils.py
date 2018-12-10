"""Build features from raw data
"""
import json
import joblib
import numpy as np
import pandas as pd
from sklearn import pipeline
from sklearn import preprocessing


with open('models/classifier.joblib', 'rb') as f:
    classifier = joblib.load(f)

with open('models/preprocessors.joblib', 'rb') as f:
    preprocessors = joblib.load(f)


COLUMN_NAMES = set(['patient_id', 'slope_of_peak_exercise_st_segment', 'thal',
                    'resting_blood_pressure', 'chest_pain_type', 'num_major_vessels',
                    'fasting_blood_sugar_gt_120_mg_per_dl', 'resting_ekg_results',
                    'serum_cholesterol_mg_per_dl', 'oldpeak_eq_st_depression', 'sex', 'age',
                    'max_heart_rate_achieved', 'exercise_induced_angina'])


def validate_content(content):
    """Validate content
    """
    missing_fields = []
    extra_fields = []
    for sample_index, sample in enumerate(content):
        columns = set(list(sample.keys()))
        missing = COLUMN_NAMES - columns
        if len(missing) > 0:
            missing_fields.append((sample_index, list(missing)))

        extra = columns - COLUMN_NAMES
        if len(extra) > 0:
            extra_fields.append((sample_index, list(extra)))

    return content, missing_fields, extra_fields


def create_dataframe(data):
    """Create a dataframe from a list of dictionaries

    Parameters
    ----------
    data : list of dict
        Each dict contains a single sample with the features described in
        https://www.drivendata.org/competitions/54/machine-learning-with-a-heart/page/109/

    Returns
    -------
    A pandas DataFrame of the input data
    """
    df = pd.DataFrame(data)
    df.set_index('patient_id', inplace=True)
    return df


def preprocess_data(data):
    """Preprocess data given a set of preprocessing steps for each variable

    Parameters
    ----------
    data : pandas.core.frame.DataFrame

    Returns
    -------
    A numpy array of size (number of rows, number of features) containing the preprocesse
    data
    """

    features = []
    for column in data:
        feat = preprocessors[column].transform(data[column].values.reshape(-1, 1))
        features.append(feat)

    return features


def concatenate_features(features):
    """Concatenate feature spaces

    Parameters
    ----------
    features : list of np.ndarray

    Returns
    -------
    An array of features with shape (number of samples, number of features)
    """
    concatenator = ConcatenateFeatures()
    _ = concatenator.fit(features)
    X = concatenator.transform(features)
    return X


class ConcatenateFeatures():
    """Translate between individual and concatenated feature spaces
    """
    def fit(self, features, names=None):
        """Build the concatenator from a list of feature spaces. A "feature space"
        is a conceptually grouped set of variables

        Parameters
        ----------
        features : list of np.ndarray
        names : list of str, optional
            If provided, the name of each feature space
        """
        if names is None:
            names = [f"feature{name}"
                     for name in range(len(features))]

        feature_sizes = []
        feature_names = []
        feature_splits = [0]
        for feature, name in zip(features, names):
            size = feature.shape[1]
            feature_sizes.append(size)
            feature_splits.append(feature_splits[-1] + size)
            feat_names = [f"{name}_{index}" for index in range(size)]
            feature_names.extend(feat_names)

        feature_splits = feature_splits[1:-1]

        self._feature_sizes = feature_sizes
        self._feature_splits = feature_splits
        self._feature_names = feature_names

        return self

    def transform(self, features):
        """Concatenate a list of feature spaces into a single feature array

        Parameters
        ----------
        features : list of np.ndarray

        Returns
        -------
        An array of features with shape (number of samples, number of features)
        """
        return np.hstack(features)

    def inverse_transform(self, features):
        """Split a feature array into a list of individual feature spaces
        """
        return np.array_split(features, self._feature_splits, axis=1)


def predict(X):
    """Use the classifier to make predictions given an array of features

    Parameters
    ----------
    X : np.ndarray
        An array of features with shape (number of samples, number of features)

    Returns
    -------
    A vector of predictions (integers, 0 or 1) and probabilities (float between 0. and 1.) for each
    sample given.
    """

    prediction = classifier.predict(X)
    probability = classifier.predict_proba(X)

    return prediction, probability

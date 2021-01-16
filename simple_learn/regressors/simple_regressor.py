# Copyright (c) 2020 Sharvil Kekre skekre98
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import json
import logging
import os
import time
import zipfile

import numpy as np
from joblib import dump, load
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.utils import all_estimators

from simple_learn.encoders import simple_model_encoder
from simple_learn.regressors.param_grid import model_param_map


class SimpleRegressor:
    """
        A class used to simplify the creation of regression models
        ...
        Attributes
        ----------
        name : str
            the optimal model algorithm for given dataset
        sk_learn : str
            the sklearn model used for prediction
        attributes : dict
            a dictionary used to keep track of model hyper-parameters
        metrics : dict
            a dictionary to keep track of scoring metrics
        gridsearch_duration : time.time
            the duration of the gridsearch being used in hyper-parameter tuning
        train_duration : time.time
            the duration of model training
        failed_models : list
            the list of failed model algorithms
        logger : logging.Logger
            logger for notifying user of warnings
        Methods
        -------
        fit(train_x, train_y, folds=3)
            Fits a given dataset onto SimpleRegressor
        predict(pred_x)
            Predicts label of samples in prediction array
        """

    def __init__(self):
        self.name = "Empty Model"
        self.sk_model = None
        self.attributes = dict()
        self.metrics = dict()
        self.gridsearch_duration = None
        self.train_duration = None
        self.failed_models = []
        self.logger = logging.getLogger()

    def __str__(self):

        for k in self.attributes:
            if type(self.attributes[k]) == np.int64:
                self.attributes[k] = int(self.attributes[k])

        attr = {
            "Type": self.name,
            "Training Duration": "{}s".format(self.train_duration),
            "GridSearch Duration": "{}s".format(self.gridsearch_duration),
            "Parameters": self.attributes,
            "Metrics": self.metrics,
        }

        str_out = json.dumps(attr, cls=simple_model_encoder.npEncoder, indent=4)
        return str_out

    def __repr__(self):

        attr = {
            "Type": self.name,
            "Training Duration": "{}s".format(self.train_duration),
            "GridSearch Duration": "{}s".format(self.gridsearch_duration),
            "Parameters": self.attributes,
            "Metrics": self.metrics,
        }

        repr_out = json.dumps(attr, cls=simple_model_encoder.npEncoder, indent=4)
        return repr_out

    def fit(self, train_x, train_y, folds=3):
        """Trains the optimal regression model
                 on given dataset by running model algorithm search.
                 If the argument folds isn't passed, the default
                 value(3) is used.
                 Parameters
                 ----------
                 train_x : numpy.ndarray
                     The features for training classification model
                 train_y : numpy.ndarray
                     The corresponding label for feature array
                 folds : int, optional
                     The number of folds for cross validation
                 """

        estimators = all_estimators(type_filter="regressor")
        for name, RegressionClass in estimators:
            if name in model_param_map:
                param_grid = model_param_map[name]
                grid_clf = GridSearchCV(
                    RegressionClass(),
                    param_grid,
                    cv=folds,
                    scoring="neg_root_mean_squared_error",
                    verbose=0,
                    n_jobs=-1,
                    error_score="raise",
                )
                start = time.time()
                try:
                    grid_clf.fit(train_x, train_y)
                except BaseException as error:
                    self.failed_models.append(name)
                    self.logger.warning(f"{name} failed due to, Error : {error}.")
                    continue
                end = time.time()
                if self.metrics.get(
                    "Training Score"
                ) is None or -grid_clf.best_score_ < self.metrics.get("Training Score"):
                    self.metrics["Training Score"] = -grid_clf.best_score_
                    pred_y = grid_clf.predict(train_x)
                    self.metrics["Mean Absolute Error"] = mean_absolute_error(
                        train_y, pred_y
                    )
                    self.metrics["Mean Square Error"] = mean_squared_error(
                        train_y, pred_y
                    )
                    self.metrics["R-Squared"] = r2_score(train_y, pred_y)
                    self.sk_model = grid_clf.best_estimator_
                    self.name = name
                    self.attributes = grid_clf.best_params_
                    self.train_duration = grid_clf.refit_time_
                    self.gridsearch_duration = end - start

    def predict(self, pred_x):
        """Predicts class label based on input
                feature array
                Parameters
                ----------
                pred_x : numpy.ndarray
                    The feature array for predicting class labels
                """
        return self.sk_model.predict(pred_x)

    def save(self, name="simple_regressor"):
        """Creates A zip archive file from SimpleRegressor
        attributes and sklearn model

        Parameters
        ----------
        name : str, optional
            The name of the zip archive file to create
        """

        dump(self.sk_model, "simple_regressor.joblib")

        clf_dict = dict()
        clf_dict["Name"] = self.name
        clf_dict["GridSearch Duration"] = self.gridsearch_duration
        clf_dict["Training Duration"] = self.train_duration
        clf_dict["Attributes"] = self.attributes
        clf_dict["Metrics"] = self.metrics
        with open("simple_regressor.json", "w") as fp:
            json.dump(clf_dict, fp)

        zip_name = "{n}.zip".format(n=name)
        zf = zipfile.ZipFile(zip_name, mode="w")
        try:
            zf.write("simple_regressor.json")
            zf.write("simple_regressor.joblib")
        finally:
            zf.close()

        os.remove("simple_regressor.json")
        os.remove("simple_regressor.joblib")

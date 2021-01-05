import numpy as np

model_param_map = {
    "SGDRegressor": {
            "loss": ["squared_loss", "huber", "epsilon_insensitive","squared_epsilon_insensitive"],
            "penalty": ["l1", "l2", "elasticnet"],
            "alpha": [0.0001,0.001],
            "learning_rate": ["constant", "optimal", "invscaling","adaptive"],
            "eta0": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2]
        },
    "KNeighborsRegressor": {
        "n_neighbors" : np.arange(3, 15),
        "weights": ["uniform", "distance"],
        "algorithm": ["ball_tree","kd_tree","brute"],
        "p": [1,2]
    },
    "DecisionTreeRegressor": {
       "criterion" : ["mse", "friedman_mse", "mae", "poisson"],
        "splitter" : ["best","random"],
        "max_depth": np.arange(3, 15),
        "max_features": ["auto", "sqrt", "log2"]
    },
    "RandomForestRegressor" : {
        "n_estimators": [200, 500],
        "criterion": ["mse","mae"],
        "max_depth": [4, 16, 32, 64, 128],
        },
    "GradientBoostingRegressor" : {
        "loss": ["ls", "lad", "huber", "quantile"],
        "learning_rate": [0.1, 0.01 , 0.001],
        "n_estimators": [50,100],
        "criterion": ["friedman_mse", "mse"],
        "max_depth": [3,6,9],
        "max_features": ["auto", "sqrt", "log2"],

    },
    "HistGradientBoostingRegressor" : {
        "loss" : ["least_squares", "least_absolute_deviation", "poisson"],
        "learning_rate": [0.01, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2],
    },
    "AdaBoostRegressor": {
        "n_estimators": [50,100],
        "loss" : ["linear", "square", "exponential"]
    }

}
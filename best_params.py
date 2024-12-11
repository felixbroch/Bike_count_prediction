parameters_LGBM = {'n_estimators': 197,
                   'learning_rate': 0.09796036713414666, 
                   'max_depth': 11, 
                   'num_leaves': 144, 
                   'min_child_samples': 26, 
                   'subsample': 0.6974141444683806, 
                   'colsample_bytree': 0.885047960342077, 
                   'reg_alpha': 0.00976316036669102, 
                   'reg_lambda': 0.2548842907160891, 
                   'tree_method': 'hist',
                   'random_state': 42
                   }



parameters_RandomForest = {
    "n_estimators": 386,
    "max_depth": 26,
    "min_samples_split": 5,
    "min_samples_leaf": 2,
    "max_features": "sqrt",
    "bootstrap": False,
    "random_state": 42,
}

parameters_XGBoost = {
    "n_estimators": 190,
    "max_depth": 6,
    "learning_rate": 0.11981406065938047,
    "subsample": 0.6544353679306312,
    "colsample_bytree": 0.6106475983764972,
    "random_state": 42,
}

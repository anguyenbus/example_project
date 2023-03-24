# External packages
import numpy as np
import lightgbm as lgb
import xgboost as xgb
import pandas as pd
from hyperopt import STATUS_OK, hp
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import (
    log_loss,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import log_loss
from sklearn.model_selection import StratifiedKFold
import xgboost as xgb
import lightgbm as lgb
from hyperopt import Trials, fmin, tpe


def lgbm_importance_plot(boost):
    """
    :info: Creates a feature importance plot from lightgbm
    :inputs:
        :model boost: The trained model
    :returns:
        :figure fig_split: feature split figure
        :figure fig_gain: feature gain figure
    """

    plot = lgb.plot_importance(
        booster=boost,
        title="Feature importance(split)",
        xlabel="Feature importance",
        ylabel="Features",
        importance_type="split",
        ignore_zero=True,
        grid=True,
        precision=3,
    )
    fig_split = plot.get_figure()

    plot = lgb.plot_importance(
        booster=boost,
        title="Feature importance(gain)",
        xlabel="Feature importance",
        ylabel="Features",
        importance_type="gain",
        ignore_zero=True,
        grid=True,
        precision=3,
    )
    fig_gain = plot.get_figure()

    return fig_split, fig_gain


def evaluate_trained_model(testev, y_test):
    """
    :info: Creates a dataframe of performance metrics for the test set. Basically a big summary of test set performance.
    :inputs:
        :pd.DataFrame testev: DataFrame of test set predictions
        :np.array y_test: yest set true labels
    :returns:
        :pd.DataFrame validation_metric: Output dataframe of validation metrics
    """

    # Set up predicted and true labels for the test set
    y_pred = testev["prediction"].values
    ytrue = y_test.values
    y_pred_binary = np.around(y_pred)

    # Create a dataframe of a whole bunch of validation metrics for the test set
    validation_metric = pd.DataFrame(columns=["metric", "value"])

    f1Score = f1_score(ytrue, y_pred_binary)
    validation_metric = validation_metric.append(
        {"metric": "f1Score", "value": f1Score}, ignore_index=True
    )

    return validation_metric


def read_predefined_parameters(ml_method):
    """
    :info: A simple container method containing default parameters for a variety of ml models
    :inputs:
        :pd.DataFrame df_features: The scikit classifier of choice
    :returns:
        :dict params: the parameters to return for the chosen model
    """

    if ml_method == "XGBoost":
        params = {
            "booster": "gbtree",
            "colsample_bytree": 0.6964465488707797,
            "gamma": 0.0,
            "learning_rate": 0.026318508413622533,
            "max_delta_step": 0,
            "max_depth": 5,
            "min_child_weight": 4.0,
            "n_jobs": -1,
            "objective": "binary:logistic",
            "reg_alpha": 0.0,
            "reg_lambda": 1.0,
            "subsample": 0.9536213610726973,
        }
    elif ml_method == "lightGBM":
        params = {
            "bagging_fraction": 0.4,
            "bagging_freq": 5,
            "boosting_type": "gbdt",
            "colsample_bytree": 0.6000000000000001,
            "early_stopping_rounds": 20,
            "importance_type": "split",
            "learning_rate": 0.06641185470713146,
            "neg_bagging_fraction": 1,
            "pos_bagging_fraction": 1,
            "max_bin": 100,
            "max_depth": 10,
            "min_child_samples": 150,
            "min_child_weight": 3,
            "min_data_in_leaf": 20,
            "min_split_gain": 0,
            "n_jobs": -1,
            "nthread": 4,
            "num_leaves": 180,
            "objective": "binary",
            "reg_alpha": 0,
            "reg_lambda": 0,
            "subsample": 0.9,
            "subsample_for_bin": 20000,
            "subsample_freq": 0,
            "verbose": 0,
        }
    return params


def perform_parameter_tuning(
    df_features, df_labels, metric_name, max_eval_param_tuning
):
    """
    :info: Sets up the Hyperopt module and begins the hyperparameter tuning process. Set
           up to work across multiple models and metrics as required.
    :inputs:
        :pd.DataFrame df_features: dataframe of features for training
        :pd.DataFrame df_labels: dataframe of labels for training
        :object metric_name: The sklearn (or other) metric to be used to perform evaluation of the model during hyper-parameter tuning
        :int max_eval_param_tuning: The maximum number of iterations to perform with hyperopt
        :int explore_dir_params: Explore output directory
    :returns:
        :dict best_params: Best params found during hyperparameter tuning
    """

    # Set up some variables for hyperopt to consume
    metric_name, MAX_EVALS, do_cv = metric_name, max_eval_param_tuning, False
    best_loss, best_params = [], []
    model_names = ["lightGBM"]
    best_params = []
    # For each choice of ml model, lets go through the hyperopt process and find the best parameters
    for model_name in model_names:
        objopt = HPOpt(df_features, df_labels, model_name, metric_name, do_cv)
        trials = Trials()
        # Get the best found parameters
        best = fmin(
            fn=objopt.do_cv_with_model,
            space=objopt.space,
            algo=tpe.suggest,
            max_evals=MAX_EVALS,
            trials=trials,
        )

        best_loss_idx = np.argmin(trials.losses())
        best_loss.append(trials.results[best_loss_idx]["loss"])
        best_params.append(trials.results[best_loss_idx]["params"])

        # best_params.append(best_params)

    # The best ml model will have the best loss result
    best_method = np.argmin(best_loss)
    print(best_method)
    # Select that method going forwards, and use the best parameters
    ml_method = model_names[best_method]
    print(best_params)
    params = best_params[best_method]
    # Setting up a dataframe of best parameters, to be appended and output every week. Probably a way neater
    # way to do this, could be re-written
    df_parameters = pd.DataFrame.from_dict(
        best_params[best_method], orient="index", columns=["optimalvalue"]
    )
    df_parameters = df_parameters.reset_index(level=0)
    df_parameters = df_parameters.rename(columns={"index": "parameters"})
    df_parameters = df_parameters.append(
        {"parameters": "optimalmethod", "optimalvalue": ml_method},
        ignore_index=True,
    )
    df_parameters = df_parameters.applymap(str)
    import pickle

    with open("best_params", "wb") as files:
        pickle.dump(df_parameters, files)

    return params


class MLTechniques:
    def xgb_teq(param, x_train, y_train, x_test, y_test, n_splits=5):
        """
        :info: Performs K-fold CV over a given training set and test set for XGBoost
               and outputs log loss metrics and prediction/true label outputs
        :inputs:
            :np.array x_train: full training set
            :np.array y_train: full training set labels
            :np.array x_test: full test set
            :np.array y_test: full test set labels
            :int n_splits: Num of K-fold splits for K-fold CV
        :returns:
            :list ts_xgb: list of log loss scores from each training set k fold split
            :list cvs_xgb: list of log loss scores from each validation set k fold split
            :float log_loss_xgboost_gradient_boosting: Entire training set log loss score
            :pd.DataFrame predictions_test_set_xgb: Single-column dataframe of test set predictions
            :pd.DataFrame preds: Two-column dataframe of test set true labels and predictions
            :object gbdt: XGBoost train object
        """

        # Read in best parameters for the model
        params_xgb = {
            "booster": param["booster"],
            "colsample_bytree": param["colsample_bytree"],
            "gamma": param["gamma"],
            "learning_rate": param["learning_rate"],
            "max_delta_step": param["max_delta_step"],
            "max_depth": param["max_depth"],
            "min_child_weight": param["min_child_weight"],
            "n_jobs": param["n_jobs"],
            "objective": param["objective"],
            "reg_alpha": param["reg_alpha"],
            "reg_lambda": param["reg_lambda"],
            "subsample": param["subsample"],
        }

        # Make a data frame of the size of y_train with one column - 'prediction'. Then make a Stratified
        # K fold object with n_splits
        ts_xgb = []
        cvs_xgb = []
        predictions_based_on_kfolds = pd.DataFrame(
            data=[], index=y_train.index, columns=["prediction"]
        )
        k_fold = StratifiedKFold(n_splits, shuffle=True)

        for train_index, cv_index in k_fold.split(
            np.zeros(len(x_train)), y_train.ravel()
        ):

            # Take subsets of X_train and y_train based on the K-fold splits
            X_train_fold, X_cv_fold = (
                x_train.iloc[train_index, :],
                x_train.iloc[cv_index, :],
            )
            y_train_fold, y_cv_fold = (
                y_train.iloc[train_index],
                y_train.iloc[cv_index],
            )

            # Convert those splits into DMatrix datasets, which are designed to interface with xgb models
            dtrain = xgb.DMatrix(data=X_train_fold, label=y_train_fold)
            dCV = xgb.DMatrix(data=X_cv_fold)

            # Use best params and the dtrain dataset to perform K fold cross validation on the (already k-folded) dataset.
            res = xgb.cv(
                params_xgb,
                dtrain,
                num_boost_round=50,
                nfold=5,
                early_stopping_rounds=10,
                verbose_eval=5,
            )
            best_nrounds = res.shape[0] - 1
            # print(np.shape(x_train), np.shape(x_test), np.shape(y_train), np.shape(y_test))

            # Train on dtrain
            gbdt = xgb.train(params_xgb, dtrain, best_nrounds)

            # Calculate training log_loss between real labels and predicted labels for this fold. Then store
            # this value in a list so that we can later take the average of all of the single fold log loss values
            log_loss_training = log_loss(y_train_fold, gbdt.predict(dtrain))
            ts_xgb.append(log_loss_training)

            # Predict on the val set and insert the predictions from this fold into the dataframe created outside the loop
            predictions_based_on_kfolds.loc[
                X_cv_fold.index, "prediction"
            ] = gbdt.predict(dCV)

            # Calculate val set log_loss between real labels and predicted labels for this fold. Then store
            # this value in a list so that we can later take the average of all of the single fold log loss values
            log_loss_cv = log_loss(
                y_cv_fold,
                predictions_based_on_kfolds.loc[X_cv_fold.index, "prediction"],
            )
            cvs_xgb.append(log_loss_cv)

        # Now that we have looped through all folds, calculate the total log loss across all data
        log_loss_xgboost_gradient_boosting = log_loss(
            y_train, predictions_based_on_kfolds.loc[:, "prediction"]
        )

        # Join up the training set labels with the predicted labels
        preds = pd.concat(
            [y_train, predictions_based_on_kfolds.loc[:, "prediction"]], axis=1
        )
        preds.columns = ["trueLabel", "prediction"]

        # Make a dmatrix out of the test set, and predict on it, then store it in a dataframe.
        dtr = xgb.DMatrix(data=x_test, label=y_test)
        predictions_test_set_xgb = pd.DataFrame(
            data=[], index=y_test.index, columns=["prediction"]
        )
        predictions_test_set_xgb.loc[:, "prediction"] = gbdt.predict(dtr)

        return (
            ts_xgb,
            cvs_xgb,
            log_loss_xgboost_gradient_boosting,
            predictions_test_set_xgb,
            preds,
            gbdt,
        )

    def lgbm_teq(param, x_train, y_train, x_test, y_test, n_splits=5):
        """
        :info: Performs K-fold CV over a given training set and test set for LightGBM
               and outputs log loss metrics and prediction/true label outputs
        :inputs:
            :np.array x_train: full training set
            :np.array y_train: full training set labels
            :np.array x_test: full test set
            :np.array y_test: full test set labels
            :int n_splits: Num of K-fold splits for K-fold CV
        :returns:
            :list ts_lightgbm: list of log loss scores from each training set k fold split
            :list cvs_lightgbm: list of log loss scores from each validation set k fold split
            :float log_loss_xgboost_gradient_boosting: Entire training set log loss score
            :pd.DataFrame predictions_test_set_xgb: Single-column dataframe of test set predictions
            :pd.DataFrame preds: Two-column dataframe of test set true labels and predictions
            :object gbm: LightGBM train object
            :int bestiteration: the best lightgbm iteration
        """

        # Read in best parameters for the model
        params_lightGB = {
            "bagging_fraction": param["bagging_fraction"],
            "objective": param["objective"],
            "bagging_freq": param["bagging_freq"],
            "boosting_type": param["boosting_type"],
            "colsample_bytree": param["colsample_bytree"],
            "neg_bagging_fraction": param["neg_bagging_fraction"],
            "pos_bagging_fraction": param["pos_bagging_fraction"],
            "importance_type": param["importance_type"],
            "learning_rate": param["learning_rate"],
            "max_depth": param["max_depth"],
            "min_child_samples": param["min_child_samples"],
            "min_data_in_leaf": param["min_data_in_leaf"],
            "max_bin": param["max_bin"],
            "min_child_weight": param["min_child_weight"],
            "subsample_for_bin": param["subsample_for_bin"],
            "nthread": param["nthread"],
            "verbose": param["verbose"],
            "early_stopping_rounds": param["early_stopping_rounds"],
            "min_split_gain": param["min_split_gain"],
            "n_jobs": param["n_jobs"],
            "num_leaves": param["num_leaves"],
            "subsample_freq": param["subsample_freq"],
            "reg_alpha": param["reg_alpha"],
            "reg_lambda": param["reg_lambda"],
            "subsample": param["subsample"],
        }

        # Make a data frame of the size of y_train with one column - 'prediction'. Then make a Stratified
        # K fold object with n_splits
        ts_lightgbm = []
        cvs_lightgbm = []
        predictions_based_on_kfolds = pd.DataFrame(
            data=[], index=y_train.index, columns=["prediction"]
        )
        k_fold = StratifiedKFold(n_splits, shuffle=True)

        for train_index, cv_index in k_fold.split(
            np.zeros(len(x_train)), y_train.ravel()
        ):

            # Take subsets of X_train and y_train based on the K-fold splits
            X_train_fold, X_cv_fold = (
                x_train.iloc[train_index, :],
                x_train.iloc[cv_index, :],
            )
            y_train_fold, y_cv_fold = (
                y_train.iloc[train_index],
                y_train.iloc[cv_index],
            )

            # Convert those splits into DMatrix datasets, which are designed to interface with lgbm models, then train
            lgb_train = lgb.Dataset(X_train_fold, y_train_fold)
            lgb_eval = lgb.Dataset(X_cv_fold, y_cv_fold, reference=lgb_train)
            gbm = lgb.train(
                params_lightGB,
                lgb_train,
                num_boost_round=1000,
                valid_sets=lgb_eval,
                early_stopping_rounds=200,
            )

            # Calculate training log_loss between real labels and predicted labels for this fold. Then store
            # this value in a list so that we can later take the average of all of the single fold log loss values
            log_loss_training = log_loss(
                y_train_fold,
                gbm.predict(X_train_fold, num_iteration=gbm.best_iteration),
            )
            ts_lightgbm.append(log_loss_training)

            # Predict on the val set and insert the predictions from this fold into the dataframe created outside the loop
            predictions_based_on_kfolds.loc[
                X_cv_fold.index, "prediction"
            ] = gbm.predict(X_cv_fold, num_iteration=gbm.best_iteration)

            # Calculate val set log_loss between real labels and predicted labels for this fold. Then store
            # this value in a list so that we can later take the average of all of the single fold log loss values
            log_loss_cv = log_loss(
                y_cv_fold,
                predictions_based_on_kfolds.loc[X_cv_fold.index, "prediction"],
            )
            cvs_lightgbm.append(log_loss_cv)

        # Now that we have looped through all folds, calculate the total log loss across all data
        log_loss_lightgbm_gradient_boosting = log_loss(
            y_train, predictions_based_on_kfolds.loc[:, "prediction"]
        )

        # Join up the training set labels with the predicted labels
        preds = pd.concat(
            [y_train, predictions_based_on_kfolds.loc[:, "prediction"]], axis=1
        )
        preds.columns = ["trueLabel", "prediction"]

        # Prepare outputs
        predictions_test_set_lightgbm = pd.DataFrame(
            data=[], index=y_test.index, columns=["prediction"]
        )
        predictions_test_set_lightgbm.loc[:, "prediction"] = gbm.predict(
            x_test, num_iteration=gbm.best_iteration
        )
        log_loss_test_set_lightgbm = log_loss(
            y_test, predictions_test_set_lightgbm
        )
        bestiteration = gbm.best_iteration

        return (
            ts_lightgbm,
            cvs_lightgbm,
            log_loss_lightgbm_gradient_boosting,
            predictions_test_set_lightgbm,
            preds,
            gbm,
            bestiteration,
        )


class HPOpt(object):
    """
    :info: The full hyperopt class. Defines variable and fixed parameter spaces, then with an input model and metric,
           trains and tests while varying over the variable parameter space. Progress is tracked using the supplied
           metric.
    :inputs:
        :np.array x_train: train-split subset of df_features as a numpy array (size determined by 1 - test_size)
        :np.array y_train: train-split subset of df_labels as a numpy array (size determined by 1 - test_size)
        :object model_name: The sklearn (or other) classifier model to be used
        :object metric_name: The sklearn (or other) metric to be used to perform evaluation of the model during hyper-parameter tuning
        :bool do_cv: Boolean variable to do K-fold cross validation or not
    """

    def __init__(self, x_train, y_train, model_name, metric_name, do_cv):
        assert model_name in [
            "lightGBM",
            "XGBoost",
            "randomForest",
            "logisticRegression",
        ], "Incorrect model chosen"
        assert metric_name in [
            "logLoss",
            "accuracyScore",
            "precisionScore",
            "recallScore",
            "f1Score",
            "ROCAUCScore",
        ], "Incorrect metric chosen"

        self.x_train = x_train
        self.y_train = y_train
        self.model_name = model_name
        self.metric_name = metric_name
        self.do_cv = do_cv

        if model_name == "lightGBM":

            ## Fixed parameters - feel free to change
            fixed_params = {
                "boosting_type": "gbdt",
                "objective": "binary",
                "bagging_freq": 5,
                "bagging_fraction": 0.4,
                "min_split_gain": 0.0,
                "reg_alpha": 0.0,
                "reg_lambda": 0.0,
                "min_data_in_leaf": 20,
                "subsample_freq": 0,  # frequence of subsample, <=0 means no enable
                "subsample_for_bin": 20000,  # Number of samples for constructing bin
                "min_split_gain": 0,  # lambda_l1, lambda_l2 and min_gain_to_split to regularization
                "reg_alpha": 0,  # L1 regularization term on weights
                "reg_lambda": 0,  # L2 regularization term on weights
                "nthread": 4,
                "verbose": 0,
                "early_stopping_rounds": 20,
                "n_jobs": -1,
            }

            ## Variable parameters - leaving n_estimators out as the train method ignores it (num_boost_rounds instead)
            variable_params = {
                "max_depth": hp.choice(
                    "max_depth", np.arange(5, 15, dtype=int)
                ),
                "num_leaves": hp.quniform("num_leaves", 100, 200, 20),
                "min_child_samples": hp.quniform(
                    "min_child_samples", 30, 180, 30
                ),
                "pos_bagging_fraction": hp.quniform(
                    "pos_bagging_fraction", 0.05, 0.95, 0.1
                ),
                "neg_bagging_fraction": hp.quniform(
                    "neg_bagging_fraction", 0.05, 0.95, 0.1
                ),
                "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
                "max_bin": hp.choice(
                    "max_bin", [50, 75, 100, 125, 150, 175, 200, 225, 250, 275]
                ),
                "colsample_bytree": hp.quniform(
                    "colsample_bytree", 0.2, 1, 0.1
                ),
                "importance_type": hp.choice(
                    "importance_type", ["split", "gain"]
                ),
                "learning_rate": hp.loguniform(
                    "learning_rate", np.log(0.01), np.log(0.2)
                ),
                "subsample": hp.quniform("subsample", 0.4, 1, 0.1),
            }

            ## Join together
            self.space = {**fixed_params, **variable_params}

        else:

            ## Fixed parameters - feel free to change
            fixed_params = {
                "booster": "gbtree",
                "gamma": 0.0,
                "max_delta_step": 0,
                "reg_alpha": 0.0,
                "reg_lambda": 1.0,
                "objective": "binary:logistic",
                "n_jobs": -1,
            }

            ## Variable parameters - leaving n_estimators out as the train method ignores it (num_boost_rounds instead)
            variable_params = {
                "learning_rate": hp.loguniform(
                    "learning_rate", np.log(0.01), np.log(0.2)
                ),
                "max_depth": hp.choice(
                    "max_depth", np.arange(1, 10, dtype=int)
                ),
                "min_child_weight": hp.quniform("min_child_weight", 1, 6, 1),
                "subsample": hp.uniform("subsample", 0.5, 1.0),
                "colsample_bytree": hp.uniform("colsample_bytree", 0.5, 1.0),
            }

            ## Join together
            self.space = {**fixed_params, **variable_params}

    ################## Metric methods ##################
    def logLoss(self, ytrue, ypred):
        return log_loss(ytrue, ypred)

    def accuracyScore(self, ytrue, ypred):
        return accuracy_score(ytrue, ypred)

    def precisionScore(self, ytrue, ypred):
        return precision_score(ytrue, ypred)

    def recallScore(self, ytrue, ypred):
        return recall_score(ytrue, ypred)

    def f1Score(self, ytrue, ypred):
        return f1_score(ytrue, ypred)

    def ROCAUCScore(self, ytrue, ypred):
        return roc_auc_score(ytrue, ypred)

    def lightGBM(self, X, y, Xval, yval, params):
        """
        :info: Train a lightGBM model on the given X, y dataset. Then predict on the training and withheld
               validation sets to get a sense of accuracy
        :inputs:
            :np.array X: feature dataset to train the model on
            :np.array y: label dataset to train the model on
            :np.array Xval: feature dataset to predict on
            :np.array yval: label dataset to predict on
            :dict params: model param dict passed in by hyperopt
        """
        dtrain = lgb.Dataset(X, y)
        dval = lgb.Dataset(Xval, yval)

        # Train the model on the training set
        trained_model = lgb.train(
            params,
            dtrain,
            num_boost_round=500,
            early_stopping_rounds=50,
            valid_sets=[dtrain, dval],
            valid_names=["train", "valid"],
        )

        # Predict on the training and val sets - this gives training and val set results
        training_set_predictions = trained_model.predict(
            X, num_iteration=trained_model.best_iteration
        )
        val_set_predictions = trained_model.predict(
            Xval, num_iteration=trained_model.best_iteration
        )
        return training_set_predictions, val_set_predictions

    def XGBoost(self, X, y, Xval, yval, params):
        """
        :info: Train an XGBoost model on the given X, y dataset. Then predict on the training and withheld
               validation sets to get a sense of accuracy
        :inputs:
            :np.array X: feature dataset to train the model on
            :np.array y: label dataset to train the model on
            :np.array Xval: feature dataset to predict on
            :np.array yval: label dataset to predict on
            :dict params: model param dict passed in by hyperopt
        """
        dtrain = xgb.DMatrix(data=X, label=y)
        dval = xgb.DMatrix(data=Xval, label=yval)

        # Train the model on the training set
        trained_model = xgb.train(
            params,
            dtrain,
            num_boost_round=5000,
            early_stopping_rounds=50,
            evals=[(dval, "valid"), (dtrain, "train")],
        )

        # Predict on the training and val sets - this gives training and val set results
        training_set_predictions = trained_model.predict(
            xgb.DMatrix(X), ntree_limit=trained_model.best_ntree_limit
        )
        val_set_predictions = trained_model.predict(
            xgb.DMatrix(Xval), ntree_limit=trained_model.best_ntree_limit
        )
        return training_set_predictions, val_set_predictions

    def logisticRegression(self, X, y, Xval, yval, params):
        """
        :info: Train a random forest model on the given X, y dataset. Then predict on the training and withheld
               validation sets to get a sense of accuracy
        :inputs:
            :np.array X: feature dataset to train the model on
            :np.array y: label dataset to train the model on
            :np.array Xval: feature dataset to predict on
            :np.array yval: label dataset to predict on
            :dict params: model param dict passed in by hyperopt
        """
        clf = LogisticRegression(params)

        # Train the model on the training set
        trained_model = clf.fit(X, y)

        # Predict on the training and val sets - this gives training and val set results
        training_set_predictions = trained_model.predict_proba(X)
        val_set_predictions = trained_model.predict_proba(Xval)
        return training_set_predictions[:, 1], val_set_predictions[:, 1]

    def do_cv_with_model(self, params):
        """
        :info: to train the supplied model on the given training data and then evaluate the supplied metric
               (accuracy, logloss, precision etc) on the validation set.
        :inputs:
            :dict params: A sampling of the parameter space (handled by hyperopt)
        :returns:
            :dict _: A dict of values including the loss, parameters and other metrics
        """

        model_name = self.model_name
        metric_name = self.metric_name

        # This is just an annoying cleaning step. lightGBM and randomForest sometimes protest that some parameters
        # aren't integers. This just forces them to be integers.
        if model_name == "lightGBM":
            intnames = ["num_leaves"]
            for ints in intnames:
                params[ints] = int(params[ints])
        elif model_name == "randomForest":
            intnames = ["n_estimators"]
            for ints in intnames:
                params[ints] = int(params[ints])

        # The metric measured on the k folds and used as the final output. Options are the above 'Score' methods
        metric = getattr(self, metric_name)

        # If do_cv, then the metric will be measured on 5 different validation sets, and averaged. If
        # do_cv == False, it will just fit on the whole training set and measure the metric on the validation set once.
        # CV takes a lot longer but will give a more accurate assessment of the metric.
        if self.do_cv:

            k_fold_training_set_metric = []
            k_fold_val_set_metric = []
            k_fold = StratifiedKFold(5, shuffle=True)

            for train_index, cv_index in k_fold.split(
                np.zeros(len(self.x_train)), self.y_train.ravel()
            ):

                # Slice x_train and y_train up according to the indices in train_index and cv_index
                X_train_fold, X_cv_fold = (
                    self.x_train.iloc[train_index, :],
                    self.x_train.iloc[cv_index, :],
                )
                y_train_fold, y_cv_fold = (
                    self.y_train.iloc[train_index],
                    self.y_train.iloc[cv_index],
                )

                # Gets the attributes of the model method here: i.e. 'lightGBM', 'XGBoost', 'randomForest' etc.
                # and then calls that method to train.
                model = getattr(self, model_name)
                training_set_predictions, val_set_predictions = model(
                    X_train_fold, y_train_fold, X_cv_fold, y_cv_fold, params
                )

                # Most scikit metrics work on (ytrue, ypreds) where ypreds are the LABELS that have been predicted.
                # However logloss requires the PROBABILITIES of the positive class (i.e. prob of a '1') rather than 0 or 1.
                if metric_name == "logLoss":
                    training_set_metric = metric(
                        y_train_fold, training_set_predictions
                    )
                    val_set_metric = metric(y_cv_fold, val_set_predictions)
                else:
                    training_set_metric = metric(
                        y_train_fold,
                        np.round(training_set_predictions).astype(int),
                    )
                    val_set_metric = metric(
                        y_cv_fold, np.round(val_set_predictions).astype(int)
                    )

                k_fold_training_set_metric.append(training_set_metric)
                k_fold_val_set_metric.append(val_set_metric)

            # Average the validation set metric results from the k folds to get an overall number
            final_training_set_metric = np.mean(k_fold_training_set_metric)
            final_val_set_metric = np.mean(k_fold_val_set_metric)

        elif self.do_cv == False:
            # If no k-fold cross-validation, then just break up into a training and validation set and measure metrics once.
            (
                X_train_fold,
                X_cv_fold,
                y_train_fold,
                y_cv_fold,
            ) = train_test_split(
                self.x_train,
                self.y_train,
                test_size=0.33,
                stratify=self.y_train,
            )

            # Gets the attributes of the model method here: i.e. 'lightGBM', 'XGBoost', 'randomForest' etc.
            # and then calls that method to train.
            model = getattr(self, model_name)
            training_set_predictions, val_set_predictions = model(
                X_train_fold, y_train_fold, X_cv_fold, y_cv_fold, params
            )

            if metric_name == "logLoss":
                final_training_set_metric = metric(
                    y_train_fold, training_set_predictions
                )
                final_val_set_metric = metric(y_cv_fold, val_set_predictions)
            else:
                final_training_set_metric = metric(
                    y_train_fold,
                    np.round(training_set_predictions).astype(int),
                )
                final_val_set_metric = metric(
                    y_cv_fold, np.round(val_set_predictions).astype(int)
                )

        # Hyperopt always MINIMISES whatever metric you give it. So if you give it something like accuracy (which
        # you want to be high) then you need to take the reciprocal
        if metric_name == "log_loss":
            loss = abs(final_val_set_metric)

        # f1 score, precison, recall are all [0, 1] - higher is better - so must take reciprocal
        elif metric_name in [
            "f1_score",
            "precision_score",
            "recall_score",
            "f1Score",
        ]:
            loss = 1.0 / final_val_set_metric

        else:
            raise Exception(
                "Have you converted the loss score such that it is going to be correctly minimised by hyperopt?"
            )

        return {
            "loss": loss,
            "training-" + metric_name: final_training_set_metric,
            "val-" + metric_name: final_val_set_metric,
            "status": STATUS_OK,
            "params": params,
        }

import numpy as np
import cv2
import sklearn.ensemble as ensemble
from sklearn.model_selection import GridSearchCV, learning_curve, cross_val_score, ShuffleSplit, train_test_split, KFold
import pandas as pd
import time

_HYPERPARAMETER_TURNING = False
_LEARNING_CURVE = False

from matplotlib import pyplot as plt

class AdaBoostRecognizer:
    def __init__(self) -> None:
        self._name = "AdaBoost Classifier"
        self._adaboost = ensemble.AdaBoostClassifier(n_estimators=50, learning_rate=0.1)

    def train(self, train_data, train_labels):
        self._train_data = train_data
        self._train_labels = train_labels

        # Split training data to training and validation set
        X_train, X_test, y_train, y_test = train_test_split(self._train_data, self._train_labels, random_state=0)

        self._split_train_data = X_train
        self._split_train_label = y_train
        self._split_val_data = X_test
        self._split_val_label = y_test

        start_training_time = time.time()

        self._adaboost.fit(self._split_train_data , self._split_train_label)

        self._training_time = time.time() - start_training_time
        print("Training took -{} seconds".format(self._training_time))

        if (_HYPERPARAMETER_TURNING):
            folds = KFold(n_splits=5, shuffle=True, random_state=5)
            hyper_parameters = [{"n_estimators": [50, 100, 200], "learning_rate": [0.1, 1.0, 2.0]}]
            boost_model = ensemble.AdaBoostClassifier()

            # Grid search
            model_cv = GridSearchCV(estimator=boost_model, param_grid=hyper_parameters, scoring="accuracy", cv=folds, verbose=1, return_train_score=True)
            model_cv.fit(self._train_data, self._train_labels)

            cv_results = pd.DataFrame(model_cv.cv_results_)
            cv_results["param_n_estimators"] = cv_results["param_n_estimators"].astype(int)

            _, axes = plt.subplots(1, 3, figsize=(20, 5))

            axes[0].set_title("Learning rate of 0.1")
            axes[0].set_xlabel("Number of estimators")
            axes[0].set_ylabel("Accuracy")
            axes[0].grid()
            lr01 = cv_results[cv_results["param_learning_rate"] == 0.1]

            axes[0].plot(lr01["param_n_estimators"], lr01["mean_test_score"])
            axes[0].plot(lr01["param_n_estimators"], lr01["mean_train_score"])
            axes[0].set_ylim(0.5, 1)
            axes[0].legend(["Test Accuracy", "Train Accuracy"], loc='upper left')

            #

            axes[1].set_title("Learning rate of 1.0")
            axes[1].set_xlabel("Number of estimators")
            axes[1].set_ylabel("Accuracy")
            axes[1].grid()
            lr10 = cv_results[cv_results["param_learning_rate"] == 1.0]

            axes[1].plot(lr10["param_n_estimators"], lr10["mean_test_score"])
            axes[1].plot(lr10["param_n_estimators"], lr10["mean_train_score"])
            axes[1].set_ylim(0.5, 1)
            axes[1].legend(["Test Accuracy", "Train Accuracy"], loc='upper left')
         
            #

            axes[2].set_title("Learning rate of 2.0")
            axes[2].set_xlabel("Number of estimators")
            axes[2].set_ylabel("Accuracy")
            axes[2].grid()
            lr_20 = cv_results[cv_results["param_learning_rate"] == 2.0]

            axes[2].plot(lr_20["param_n_estimators"], lr_20["mean_test_score"])
            axes[2].plot(lr_20["param_n_estimators"], lr_20["mean_train_score"])
            axes[2].set_ylim(0.5, 1)
            axes[2].legend(["Test Accuracy", "Train Accuracy"], loc='upper left')
            plt.savefig("AdaBoost Tuning.png")

            best_score = model_cv.best_score_
            best_hyperparameters = model_cv.best_params_

            print("Best test score -{0}, best hyperparameters {1}".format(best_score, best_hyperparameters))

    def predict(self, test_data, test_labels): 
        val_score = self._adaboost.score(self._split_val_data, self._split_val_label)

        start_test_time = time.time()
    
        test_score = self._adaboost.score(test_data, test_labels)

        self._test_time = time.time() - start_test_time
        print("Testing took -{} seconds".format(self._test_time))

        print("{} val accuracy: {}".format(self._name, val_score))
        print("{} testing accuracy: {}".format(self._name, test_score))
        
    def plot_learning_curve(self):
        if (_LEARNING_CURVE):
            _, axes = plt.subplots(1, 3, figsize=(20, 5))

            axes[0].set_title(self._name)
            axes[0].set_xlabel("Training examples")
            axes[0].set_ylabel("Score")

            cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

            train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
                self._adaboost,
                self._train_data,
                self._train_labels,
                cv=cv,
                n_jobs=4,
                return_times=True,
            )

            train_scores_mean = np.mean(train_scores, axis=1)
            train_scores_std = np.std(train_scores, axis=1)
            test_scores_mean = np.mean(test_scores, axis=1)
            test_scores_std = np.std(test_scores, axis=1)
            fit_times_mean = np.mean(fit_times, axis=1)
            fit_times_std = np.std(fit_times, axis=1)

            # Plot learning curve
            axes[0].grid()
            axes[0].fill_between(
                train_sizes,
                train_scores_mean - train_scores_std,
                train_scores_mean + train_scores_std,
                alpha=0.1,
                color="r",
            )
            axes[0].fill_between(
                train_sizes,
                test_scores_mean - test_scores_std,
                test_scores_mean + test_scores_std,
                alpha=0.1,
                color="g",
            )
            axes[0].plot(
                train_sizes, train_scores_mean, "o-", color="r", label="Training score"
            )
            axes[0].plot(
                train_sizes, test_scores_mean, "o-", color="g", label="Cross-validation score"
            )
            axes[0].legend(loc="best")

            # Plot n_samples vs fit_times
            axes[1].grid()
            axes[1].plot(train_sizes, fit_times_mean, "o-")
            axes[1].fill_between(
                train_sizes,
                fit_times_mean - fit_times_std,
                fit_times_mean + fit_times_std,
                alpha=0.1,
            )
            axes[1].set_xlabel("Training examples")
            axes[1].set_ylabel("fit_times")
            axes[1].set_title("Scalability of the model")

            # Plot fit_time vs score
            fit_time_argsort = fit_times_mean.argsort()
            fit_time_sorted = fit_times_mean[fit_time_argsort]
            test_scores_mean_sorted = test_scores_mean[fit_time_argsort]
            test_scores_std_sorted = test_scores_std[fit_time_argsort]
            axes[2].grid()
            axes[2].plot(fit_time_sorted, test_scores_mean_sorted, "o-")
            axes[2].fill_between(
                fit_time_sorted,
                test_scores_mean_sorted - test_scores_std_sorted,
                test_scores_mean_sorted + test_scores_std_sorted,
                alpha=0.1,
            )
            axes[2].set_xlabel("fit_times")
            axes[2].set_ylabel("Score")
            axes[2].set_title("Performance of the model")

            plt.savefig("{}.png".format(self._name))     

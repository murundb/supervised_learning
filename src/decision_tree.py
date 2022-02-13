import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import learning_curve, cross_val_score, ShuffleSplit, train_test_split

_VISUALIZE_IMAGES = False
_TREE_DEPTH_EVALUATION = False
_TREE_COST_COMPLEXITY_ANALYSIS = False

_NN_EVALUATION = True

from matplotlib import pyplot as plt

class DigitRecognizers:
    """
    Class that encapsulates the digit recognizer algorithms
    """

    def __init__(self, type="DecisionTree", dataset=None, datalabels=None) -> None:
        self._train_data = dataset
        self._train_labels = datalabels
        self._train_features = generate_hog_features(self._train_data)

        self._clfs = list()
        #self._clfs.append(DecisionTreeDigitRecognizer())
        self._clfs.append(NeuralNetworkDigitRecognizer())

        # if type == "all":
        #     self._clfs.append(DecisionTreeDigitRecognizer())
        #     self._clfs.append(NeuralNetworkDigitRecognizer())
        # elif type == "decision_tree":
        #     self._clfs.append(DecisionTreeDigitRecognizer())
        # elif type == "neural_network":
        #     self._clfs.append(NeuralNetworkDigitRecognizer())

    def train(self):
        for clf in self._clfs:
            clf.train(self._train_features, self._train_labels)

    def predict(self, test_data, test_labels):
        test_features = generate_hog_features(test_data)
        for clf in self._clfs:
            clf.predict(test_features, test_labels)

    def plot_learning_curves(self):
        for clf in self._clfs:
            clf.plot_learning_curve()


class DecisionTreeDigitRecognizer:
    def __init__(self) -> None:
        self._name = "Decision Tree Classifier"
        self._dtree = DecisionTreeClassifier(criterion="gini", splitter="best", ccp_alpha=0.005)
        if (_TREE_DEPTH_EVALUATION):
            self._num_tree = 10
            _start_tree_depth = 5
            self.dtree_eval_list = list()

            for i in range(self._num_tree):
                self.dtree_eval_list.append(DecisionTreeClassifier(criterion="gini", splitter="best", max_depth=_start_tree_depth))
                _start_tree_depth += 5
        
    def train(self, train_data, train_labels):

        self._train_data = train_data
        self._train_labels = train_labels
        self._dtree.fit(train_data , train_labels)

    def predict(self, test_data, test_labels): 
        train_score = self._dtree.score(self._train_data, self._train_labels)
        test_score = self._dtree.score(test_data, test_labels)

        print("{} training accuracy: {}".format(self._name, train_score))
        print("{} testing accuracy: {}".format(self._name, test_score))

        if (_TREE_DEPTH_EVALUATION):
            scores = list()
            tree_depth = list()
            _start_tree_depth = 5
            for i in range(self._num_tree):
                score = cross_val_score(self.dtree_eval_list[i], self._train_data, self._train_labels, cv=5, scoring="accuracy")
                scores.append(score.mean())
                tree_depth.append(_start_tree_depth)
                _start_tree_depth += 5
            
            fig, ax = plt.subplots()
            ax.set_title("Decision Tree Cross-Validation Error")
            ax.set_xlabel("Maximum Tree Depth")
            ax.set_ylabel("Accuracy")
            ax.plot(tree_depth, scores, '-o')
            plt.savefig("Decision Tree Depth.png")

        if (_TREE_COST_COMPLEXITY_ANALYSIS):
            X_train, X_test, y_train, y_test = train_test_split(self._train_data, self._train_labels, random_state=0)
            clf = DecisionTreeClassifier(random_state=0)
            path = clf.cost_complexity_pruning_path(X_train, y_train)
            ccp_alphas, impurities = path.ccp_alphas, path.impurities

            clfs = []
            for ccp_alpha in ccp_alphas:
                clf = DecisionTreeClassifier(random_state=0, ccp_alpha=ccp_alpha)
                clf.fit(X_train, y_train)
                clfs.append(clf)
            print(
                "Number of nodes in the last tree is: {} with ccp_alpha: {}".format(
                    clfs[-1].tree_.node_count, ccp_alphas[-1]
                )
            )

            clfs = clfs[:-1]
            ccp_alphas = ccp_alphas[:-1]

            train_scores = [clf.score(X_train, y_train) for clf in clfs]
            test_scores = [clf.score(X_test, y_test) for clf in clfs]

            fig, ax = plt.subplots()
            ax.set_xlabel("alpha")
            ax.set_ylabel("accuracy")
            ax.set_title("Accuracy vs alpha for training and validation sets")
            ax.plot(ccp_alphas, train_scores, marker="o", label="train", drawstyle="steps-post")
            ax.plot(ccp_alphas, test_scores, marker="o", label="test", drawstyle="steps-post")
            ax.legend()

            plt.savefig("effective_alpha.png")

    def plot_learning_curve(self):
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

        axes[0].set_title(self._name)
        axes[0].set_xlabel("Training examples")
        axes[0].set_ylabel("Score")

        cv = ShuffleSplit(n_splits=5, test_size=0.2, random_state=0)

        train_sizes, train_scores, test_scores, fit_times, _ = learning_curve(
            self._dtree,
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

class NeuralNetworkDigitRecognizer:
    def __init__(self) -> None:
        self._name = "Neural Network Classifier"
        self._nn = MLPClassifier(solver='adam', 
                    activation='logistic', 
                    alpha=.001, 
                     hidden_layer_sizes=(512, 128, 10), 
                    #hidden_layer_sizes=(784, 196, 10), 
                    random_state=1, 
                    max_iter=500)

    def train(self, train_data, train_labels):
        self._train_data = train_data
        self._train_labels = train_labels
        self._nn.fit(self._train_data, self._train_labels)
    
    def predict(self, test_data, test_labels): 
        pred_train = self._nn.predict(self._train_data)
        train_score = np.mean(pred_train == self._train_labels)

        # Evaluate
        pred_test = self._nn.predict(test_data)
        test_score = np.mean(pred_test == test_labels)

        print("{} training accuracy: {}".format(self._name, train_score))
        print("{} testing accuracy: {}".format(self._name, test_score))

    def plot_learning_curve(self):
        pass

class KNearestNeighboursRecognizer:
    def __init__(self) -> None:
        self._name = "K-Nearest-Neighbours Classifier"
        self._nn = MLPClassifier(solver='adam', 
                    activation='relu', 
                    alpha=.001, 
                    hidden_layer_sizes=(512, 128, 10), 
                    random_state=1, 
                    max_iter=500)
    
    def train(self, train_data, train_labels):
        self._train_data = train_data
        self._train_labels = train_labels
        self._nn.fit(self._train_data, self._train_labels)
    
    def predict(self, test_data, test_labels): 
        pred_train = self._nn.predict(self._train_data)
        train_score = np.mean(pred_train == self._train_labels)

        # Evaluate
        pred_test = self._nn.predict(test_data)
        test_score = np.mean(pred_test == test_labels)

        print("{} training accuracy: {}".format(self._name, train_score))
        print("{} testing accuracy: {}".format(self._name, test_score))
    

def generate_hog_features(dataset):
    """
    Generates HOG features
    """
    hog_list = []
    hog_image_list = []
    for i in range(dataset.shape[0]):
        img = dataset[i]
        feature, hog_image = hog(img, orientations=9, pixels_per_cell=(4, 4), cells_per_block=(3, 3), visualize=True)
        hog_list.append(feature)
        hog_image_list.append(hog_image)

    hog_image_list = np.array(hog_image_list)

    if (_VISUALIZE_IMAGES):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

        ax1.axis('off')
        ax1.imshow(dataset[9], cmap=plt.cm.gray)
        ax1.set_title('MNIST Dataset - Digit 9')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image_list[9], in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.savefig("MNIST.png")  

    return np.array(hog_list)

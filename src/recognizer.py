import numpy as np
import cv2
from skimage.feature import hog
from skimage import exposure

from src.decision_tree import DecisionTreeDigitRecognizer
from src.artificial_neural_network import ArtificialNeuralNetworkRecognizer
from src.suppor_vector_machine import SupportVectorMachineRecognizer
from src.k_nearest_neighbours import KNearestNeighboursRecognizer
from src.adaboost import AdaBoostRecognizer


_VISUALIZE_IMAGES = False
_TREE_DEPTH_EVALUATION = False
_TREE_COST_COMPLEXITY_ANALYSIS = False

_NN_EVALUATION = True

from matplotlib import pyplot as plt

class DigitRecognizers:
    """
    Class that encapsulates the digit recognizer algorithms
    """

    def __init__(self, type="decision_tree", dataset=None, datalabels=None) -> None:
        self._train_data = dataset
        self._train_labels = datalabels
        self._train_features = generate_hog_features(self._train_data)

        self._clfs = list()

        if (type == "dt"):
            self._clfs.append(DecisionTreeDigitRecognizer())
        elif (type == "nn"):
            self._clfs.append(ArtificialNeuralNetworkRecognizer())
        elif (type == "svm"):
            self._clfs.append(SupportVectorMachineRecognizer())
        elif (type == "adaboost"):
            self._clfs.append(AdaBoostRecognizer())
        elif (type == "knn"):
            self._clfs.append(KNearestNeighboursRecognizer())
        else:
            self._clfs.append(DecisionTreeDigitRecognizer())
            self._clfs.append(ArtificialNeuralNetworkRecognizer())
            self._clfs.append(SupportVectorMachineRecognizer())
            self._clfs.append(KNearestNeighboursRecognizer())
            self._clfs.append(AdaBoostRecognizer())

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
        ax1.set_title('Asirra Dataset - Digit 9')

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image_list[9], in_range=(0, 10))

        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')
        plt.savefig("Assra.png")  

    return np.array(hog_list)

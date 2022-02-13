import os, sys
import click
import cv2
import torchvision.datasets as datasets
from matplotlib import pyplot as plt
import numpy as np
from src.recognizer import DigitRecognizers

USE_ASIRRA_DATASET= True
USE_MNIST_DATASET = False

K_NUMBER_OF_TRAINING_DATASET = 5000
K_NUMBER_OF_TEST_DATASET = 10000

@click.command()
@click.option("--type", required=False, type=str)
def main(type):

    if (USE_MNIST_DATASET):
        # https://pytorch.org/vision/stable/datasets.html
        # http://yann.lecun.com/exdb/mnist/
        mnist_trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
        mnist_testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)

        # Train set
        train_data = np.array(mnist_trainset.data)[:K_NUMBER_OF_TRAINING_DATASET]
        train_labels = np.array(mnist_trainset.targets)[:K_NUMBER_OF_TRAINING_DATASET]

        # Test set
        test_data = np.array(mnist_testset.data)[:K_NUMBER_OF_TEST_DATASET]
        test_labels = np.array(mnist_testset.targets)[:K_NUMBER_OF_TEST_DATASET]
    
    elif (USE_ASIRRA_DATASET):
        path_to_train_data = "./data/Asirra/train"
        path_to_test_data = "./data/Asirra/test"

        filenames_train = os.listdir(path_to_train_data)
        filenames_test = os.listdir(path_to_test_data)

        train_data = load_asirra(path_to_train_data, 400)        
        test_data = load_asirra(path_to_test_data, 2000)

        train_labels = np.zeros(400, dtype=int)
        test_labels = np.zeros(2000, dtype=int)

        i = 0
        for filename in filenames_train:
            label = filename.split(".")[0]
            if label == "dog":
                train_labels[i] = 1
                i += 1
            elif label == "cat":
                train_labels[i] = 0
                i += 1

        j = 0
        for filename in filenames_test:
            label = filename.split(".")[0]
            if label == "dog":
                test_labels[j] = 1
                j += 1
            elif label == "cat":
                test_labels[j] = 0
                j += 1
    else:
        sys.exit()
        
    # Create a classifier
    recognizers = DigitRecognizers(type=type, dataset=train_data, datalabels=train_labels)

    recognizers.train()

    # Evaluate on train-test dataset
    recognizers.predict(test_data, test_labels)

    # Plot learning curves
    # https://scikit-learn.org/stable/auto_examples/model_selection/plot_learning_curve.html#sphx-glr-auto-examples-model-selection-plot-learning-curve-py
    recognizers.plot_learning_curves()

def load_asirra(path_to_data, data_length):
    images = np.zeros((data_length, 200, 200), dtype=np.uint8)

    cnt = 0
    for filename in os.listdir(path_to_data):
        img = cv2.imread(os.path.join(path_to_data, filename), cv2.IMREAD_GRAYSCALE)

        if img is not None:
            resized = cv2.resize(img, (200, 200), interpolation = cv2.INTER_AREA)
            images[cnt, :, :] = resized
            cnt +=1 

        

    return images

if __name__ == "__main__":
    main()

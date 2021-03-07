"""
mpg (Miles per Gallon) Predictor using OLS (Ordinary Linear Regression) 

File: ECE5268_MP1_P2.py
Author: Richard Hemphill
ID: 903877709
Class: ECE5268 Theory of Neural Networks
Teacher: Dr. Georgios C. Anagnostopoulos
Project: Consider the dataset provided in autompg_dataset.csv, which was derived from the Auto MPG Data Set dataset 
hosted at the UCI ML Repository by removing samples with missing feature values. We are going to use this dataset to 
predict a car’s miles-pergallon fuel consumption mpg using OLS regression based on a number of car characteristics.
First, split the available data into a training set (first 200 samples), a validation set (the next 100 samples) and a 
test set (remaining 92 samples).
    Part (a): By minimizing the training MSE, fit a linear regression model with intercept (let’s call it Model1) that 
    predicts mpg based on the horsepower and weight input features. Report (i) the fitted model’s prediction equation 
    and (ii) the optimal training MSE value achieved. Does the prediction equation make intuitive sense or not? 
    Justify your answer.
    Part (b): Produce a plot showing the fitted model’s response surface (it will be a plane) superimposed on the 
    scatter plot of the training data (horsepower, weight, mpg). Comment on the obtained plot.
    Part (c): Repeat part (a), but, this time, use all available input features (except for model year, origin, and 
    car name). Let’s call this model Model2. Compare the obtained results to the ones of part (a).
    Part (d): Select the best of Model1 and Model2 based on the validation MSE. Finally, for the best model, report 
    its test MSE. Provide pertinent commentary on your findings.
"""

# Libraries
import numpy as np
import random
import matplotlib.pyplot as plt

class OlsModel(object):

    # Constants
    DATASET_FILE = 'autompg_dataset.csv'

    def __init__(self, name, inputList, output, shuffle=True, dataSetFile=DATASET_FILE):
        """ Constructor """
        # dataSetFile - CSV file containing automotive data.
        self._name = name
        self._inputList = inputList
        self._output = output
        self._shuffle = shuffle
        self._dataSetFile = dataSetFile
        self._data = []
        self._X = []
        self._Y = []
        self._W = []
        self._loadDataSet()
        self._initData()

    @property
    def name(self):
        return self._name

    def split(self, train, validation):
        self._Xtrain = self._X[:train-1]
        self._Ytrain = self._Y[:train-1]
        self._Xvalidation = self._X[train:train+validation-1]
        self._Yvalidation = self._Y[train:train+validation-1]
        self._Xtest = self._X[validation:]
        self._Ytest = self._Y[validation:]

    def train(self, report=False):
        self._R = np.dot(self._Xtrain.T, self._Xtrain)
        self._Rinv = np.linalg.inv(self._R)
        self._W = np.dot(np.dot(self._Rinv, self._Xtrain.T), self._Ytrain)
        mse = self._mse(self._Ytrain, np.dot(self._Xtrain, self._W))
        if report:
            print('Training MSE ({}): {:0.6}'.format(self.name, mse))
        return mse

    def validate(self, report=False):
        mse = self._mse(self._Yvalidation, np.dot(self._Xvalidation, self._W))
        if report:
            print('Validation MSE ({}): {:0.6}'.format(self.name, mse))
        return mse

    def test(self, report=False):
        mse = self._mse(self._Ytest, np.dot(self._Xtest, self._W))
        if report:
            print('Test MSE ({}): {:0.6}'.format(self.name, mse))
        return mse

    def plot(self, show=False):
        numSteps = 100
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        Xmin = min(self._X[:,0])
        Xmax = max(self._X[:,0])
        Xstep = (Xmax-Xmin)/numSteps
        x = np.arange(Xmin, Xmax, Xstep)
        Ymin = min(self._X[:,1])
        Ymax = max(self._X[:,1])
        Ystep = (Ymax - Ymin)/numSteps
        y = np.arange(Ymin, Ymax, Ystep)
        X, Y = np.meshgrid(x, y)
        zs = np.array(self._preditionEquation(np.ravel(X), np.ravel(Y)))
        Z = zs.reshape(X.shape)

        ax.plot_surface(X, Y, Z)
        ax.scatter(self._X[:,0], self._X[:,1], self._Y)

        ax.set_xlabel(self._inputList[0])
        ax.set_ylabel(self._inputList[1])
        ax.set_zlabel(self._output)
        plt.title(self._name)
        plt.savefig('{}.jpg'.format(self._name))
        if show:
            plt.show()

    def __del__(self):
        """ Descructor """
        self._data = None
        self._X = None

    def __str__(self):
        eq = 'Prediction Equation ({}): {} = '.format(self._name, self._output)
        weight = lambda i: ('+' if i > 0 else '') + '{:0.6}'.format(i)
        for idx, x in enumerate(self._inputList):
            eq = eq + '{}*{}'.format(weight(self._W[idx]), x)
        eq = eq + weight(self._W[-1])
        return eq

    def _loadDataSet(self):
        """ Loads cata into data matrix """
        # dataSetFile - CSV file containing automotive data.

        # retrieve data from CSV
        csvFile = open(self._dataSetFile, 'r')
        self._data = np.genfromtxt(csvFile, delimiter=',', names=True, case_sensitive=True)
        csvFile.close()

        # shuffle data randomly so that training will not use same sets
        if self._shuffle:
            random.shuffle(self._data)

    def _initData(self):
        """ Initialize matrices used for regression """
        # create design matrix with list of input parameters
        self._X = self._data[self._inputList[0]]
        for input in self._inputList[1:]:
            self._X = np.column_stack((self._X, self._data[input]))

        # column of ones to augment the design matrix
        self._X = np.column_stack((self._X, np.ones(len(self._X))))

        # create output vector from data
        self._Y = self._data[self._output]

        self._W = np.ones(len(self._inputList)+1)

    def _preditionEquation(self, x, y):
        return self._W[0] * x + self._W[1] * y + self._W[2]

    def _mse(self, actual, predicted):
        return np.square(np.subtract(actual, predicted)).mean()


def determineBest(models):
    bestMse = float("inf")
    for model in models:
        mse = model.validate(report=True)
        if mse < bestMse:
            bestMse = mse
            bestModel = model

    print('{} is the best and has a test set MSE of {:0.6}!'.format(bestModel.name, bestModel.test()))

def processModels(models, train, validation):
    for model in models:
        model.split(train, validation)
        model.train(report=True)
        print(model)

def main():
    shuffleData=True
    models = []
    models.append(OlsModel(name='model1',inputList=['cylinders', 'displacement'],output='mpg',shuffle=shuffleData))
    models.append(OlsModel(name='model2', inputList=['cylinders', 'displacement', 'horsepower', 'weight','acceleration'],output='mpg',shuffle=shuffleData))
    processModels(models, train=200, validation=100)
    determineBest(models)
    models[0].plot(show=True)


if __name__ == "__main__":
    main()
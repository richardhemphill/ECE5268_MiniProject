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
import csv
import numpy as np
import matplotlib.pyplot as plt

class MpgModel(object):

    # Constants
    DATASET_FILE = 'autompg_dataset.csv'

    def __init__(self, dataSetFile=DATASET_FILE):
        """ Constructor """
        # dataSetFile - CSV file containing automotive data.
        self._loadDataSet(dataSetFile)

    def __del__(self):
        """ Descructor """
        self._data = None

    def __str__(self):
        return repr(self._data)

    def _loadDataSet(self, dataSetFile):
        """ Loads Data into Data Matrix """
        # dataSetFile - CSV file containing automotive data.
        csvFile = open(dataSetFile, 'r')
        #self._data = list(csv.reader(csvFile, delimiter=","))
        self._data = np.genfromtxt(csvFile, delimiter=',', names=True, case_sensitive=True)
        csvFile.close()

def main():
    mpgModel = MpgModel()
    print(mpgModel)

if __name__ == "__main__":
    main()
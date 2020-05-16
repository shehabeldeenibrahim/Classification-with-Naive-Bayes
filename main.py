# Import libraries
import numpy as np
import pandas as pd
import seaborn as sn
import math
from sklearn import metrics 
import matplotlib.pyplot as plt


def prior_probability(class0, class1):
    # Compute prior probability for each class
    total = class0.shape[0] + class1.shape[0]
    prior0 = class0.shape[0] / total
    prior1 = class1.shape[0] / total
    return prior0, prior1

def replace_zeroes(arr, n):
    # Replace 0 with 0.0001 in array
    for i in range(len(arr)):
        if(arr[i] <= 0):
            arr[i] = n
    return arr

def mean_std(class0, class1):
    # Compute mean and std deviation for each class and feature
    mean0 = []
    mean1 = []
    stdDev0 = []
    stdDev1 = []

    for i in range(class0.shape[1]):
        mean0.append(np.mean(class0[:,i]))
        stdDev0.append(np.std(class0[:,i]))


    for i in range(class1.shape[1]):
        mean1.append(np.mean(class1[:,i]))  
        stdDev1.append(np.std(class1[:,i]))  

    stdDev0 = replace_zeroes(stdDev0, 0.001)
    stdDev1 = replace_zeroes(stdDev1, 0.001)

    return mean0, stdDev0, mean1, stdDev1

def naive_bayes(prior0, prior1, mean0, stdDev0, mean1, stdDev1, testClass0, testClass1):
    # Predict classes for test cases given and calculate the accuracy, 
    # precision and recall

    p0 = []
    p1 = []
    finalPredictions0 = []
    finalPredictions1 = []
    prediction = [0,0]

    # Run Naive Bayes on the 0 test cases
    for i in range(len(testClass0)):
        p0 = []
        p1 = []
        for j in range(len(mean0)):

            normal0 = N(testClass0[i][j], mean0[j], stdDev0[j])
            normal1 = N(testClass0[i][j], mean1[j], stdDev1[j])

            p0.append(normal0)
            p1.append(normal1)
        prediction[0] = np.prod(p0) * (prior0)
        prediction[1] = np.prod(p1) * (prior1)
        finalPredictions0.append(np.argmax(prediction))

    # Run Naive Bayes on the 1 test cases
    for i in range(len(testClass1)):
        p0 = []
        p1 = []
        for j in range(len(mean0)):

            normal0 = N(testClass1[i][j], mean0[j], stdDev0[j])
            normal1 = N(testClass1[i][j], mean1[j], stdDev1[j])

            p0.append(normal0)
            p1.append(normal1)
        prediction[0] = np.prod(p0) * (prior0)
        prediction[1] = np.prod(p1) * (prior1)
        finalPredictions1.append(np.argmax(prediction))

    labels = np.concatenate((np.zeros(len(finalPredictions0)), np.ones(len(finalPredictions1))))
    finalPredictions = np.concatenate((finalPredictions0,finalPredictions1))
    
    return labels, finalPredictions

def N(x, mean, stdDev):
    # Calculate the normal dis. for a given input x
	exponent = math.exp(-((x-mean)**2 / (2 * stdDev**2 )))
	return (1 / (math.sqrt(2 * math.pi) * stdDev)) * exponent

def confusion_matrix(labels, finalPredictions):
    # Create confusion matrix
    confusionMatrix = metrics.confusion_matrix(labels ,finalPredictions)
    print(confusionMatrix)
    df_cm = pd.DataFrame(confusionMatrix, index = [i for i in "01"],
                    columns = [i for i in "01"])
    plt.figure(figsize = (10,7))
    sn.heatmap(df_cm, annot=True, fmt='g')

    # Plot accuracy graph and confusion matrix
    plt.show()

# Import dataset and seperate into two classes
datasetTrain = pd.read_csv('dataset/spambase.data')
datasetValues = datasetTrain.values
class0 = datasetValues[datasetValues[:,57] == 0]
class1 = datasetValues[datasetValues[:,57] == 1]

# Split dataset to training and test with 40% spam and 60% non spam
trainingClass0 = class0[:1380, :-1]
trainingClass1 = class1[:920, :-1]

testClass0 = class0[1380:, :-1]
testClass1 = class1[920:, :-1]


# Prior probability for each class in the training
prior0, prior1 = prior_probability(trainingClass0, trainingClass1)

# Mean and std dev for each training class, 57 feature 
mean0, stdDev0, mean1, stdDev1 = mean_std(trainingClass0, trainingClass1)

# Run Naive Bayes on testset
labels, finalPredictions = naive_bayes(prior0, prior1, mean0, stdDev0, mean1, stdDev1, testClass0, testClass1)

# Run analysis
print("Gaussian Naive Bayes model accuracy:", metrics.accuracy_score(labels, finalPredictions))
print("Gaussian Naive Bayes model precision:", metrics.precision_score(labels, finalPredictions))
print("Gaussian Naive Bayes model recall:", metrics.recall_score(labels, finalPredictions))
confusion_matrix(labels, finalPredictions)

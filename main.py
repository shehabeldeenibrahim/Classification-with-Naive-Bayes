# Import libraries
import numpy as np
import pandas as pd
import math

def prior_probability(class0, class1):
    # Compute prior probability for each class
    total = class0.shape[0] + class1.shape[0]
    prior0 = class0.shape[0] / total
    prior1 = class1.shape[0] / total
    return prior0, prior1

def replace_zeroes(arr):
    # Replace 0 with 0.0001 in array
    for i in range(len(arr)):
        if(arr[i] == 0):
            arr[i] = 0.0001
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

    stdDev0 = replace_zeroes(stdDev0)
    stdDev1 = replace_zeroes(stdDev1)

    return mean0, stdDev0, mean1, stdDev1

def naive_bayes(prior0, prior1, mean0, stdDev0, mean1, stdDev1, testClass0, testClass1):
    p0 = []
    p1 = []
    finalPredictions0 = []
    finalPredictions1 = []
    prediction = [0,0]
    for i in range(len(testClass0)):
        for j in range(len(mean0)):

            normal0 = N(testClass0[i][j], mean0[j], stdDev0[j])
            normal0 = normal0 if normal0 > 0.0 else 0.0001
            normal1 = N(testClass0[i][j], mean1[j], stdDev1[j])
            normal1 = normal1 if normal1 > 0.0 else 0.0001

            p0.append(math.log(normal0))
            p1.append(math.log(normal1))
        prediction[0] = sum(p0) + math.log(prior0)
        prediction[1] = sum(p1) + math.log(prior1)
        finalPredictions0.append(np.argmax(prediction))

    for i in range(len(testClass1)):
        for j in range(len(mean0)):

            normal0 = N(testClass1[i][j], mean0[j], stdDev0[j])
            normal0 = normal0 if normal0 > 0.0 else 0.0001
            normal1 = N(testClass1[i][j], mean1[j], stdDev1[j])
            normal1 = normal1 if normal1 > 0.0 else 0.0001

            p0.append(math.log(normal0))
            p1.append(math.log(normal1))
        prediction[0] = sum(p0) + math.log(prior0)
        prediction[1] = sum(p1) + math.log(prior1)
        finalPredictions1.append(np.argmax(prediction))

    nInstances = testClass0.shape[0] + testClass1.shape[0]
    accuracy0 = finalPredictions0.count(0) / len(finalPredictions0)
    accuracy1 = finalPredictions1.count(1) / len(finalPredictions1)
    
    return  

def N(x, mean, stdDev):
	exponent = math.exp(-((x-mean)**2 / (2 * stdDev**2 )))
	return (1 / (math.sqrt(2 * math.pi) * stdDev)) * exponent

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

trainingSet = np.concatenate((trainingClass0, trainingClass1))
testSet = np.concatenate((testClass0, testClass1))

trainingClass1 = np.array(([3.0, 5.1], [4.1, 6.3], [7.2, 9.8]))
trainingClass0 = np.array(([2.0, 1.1], [4.1, 2.0], [8.1, 9.4]))
testClass0 = np.array(([5.2,6.3],[5.2,6.3]))
testClass1 = np.array(([5.2,6.3],[5.2,6.3]))

# Prior probability for each class in the training
prior0, prior1 = prior_probability(trainingClass0, trainingClass1)

# Mean and std dev for each training class, 57 feature 
mean0, stdDev0, mean1, stdDev1 = mean_std(trainingClass0, trainingClass1)

# Run Naive Bayes on testset
naive_bayes(prior0, prior1, mean0, stdDev0, mean1, stdDev1, testClass0, testClass1)
print("")
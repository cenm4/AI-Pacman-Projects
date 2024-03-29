# runMinicontest.py
# -----------------
# Licensing Information: Please do not distribute or publish solutions to this
# project. You are free to use and extend these projects for educational
# purposes. The Pacman AI projects were developed at UC Berkeley, primarily by
# John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and Pieter 
# Abbeel in Spring 2013.
# For more info, see http://inst.eecs.berkeley.edu/~cs188/pacman/pacman.html

# This file is for running the minicontest submission.

import minicontest
import samples
import sys
import util
import pickle
from dataClassifier import DIGIT_DATUM_HEIGHT,DIGIT_DATUM_WIDTH,contestFeatureExtractorDigit

TEST_SIZE = 1000

MINICONTEST_PATH = "minicontest_output.pickle"


if __name__ == '__main__':
    print("Loading training data")
    rawTrainingData = samples.loadDataFile("data/digitdata/trainingimages", 5000,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    trainingLabels = samples.loadLabelsFile("data/digitdata/traininglabels", 5000)
    rawValidationData = samples.loadDataFile("data/digitdata/validationimages", 100,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)
    validationLabels = samples.loadLabelsFile("data/digitdata/validationlabels", 100)
    rawTestData = samples.loadDataFile("data/digitdata/testimages", TEST_SIZE,DIGIT_DATUM_WIDTH,DIGIT_DATUM_HEIGHT)


    featureFunction = contestFeatureExtractorDigit
    legalLabels = list(range(10))
    classifier = minicontest.contestClassifier(legalLabels)

    print("Extracting features...")
    trainingData = list(map(featureFunction, rawTrainingData))
    validationData = list(map(featureFunction, rawValidationData))
    testData = list(map(featureFunction, rawTestData))

    print("Training...")
    classifier.train(trainingData, trainingLabels, validationData, validationLabels)
    print("Validating...")
    guesses = classifier.classify(validationData)
    correct = [guesses[i] == validationLabels[i] for i in range(len(validationLabels))].count(True)
    print(str(correct), ("correct out of " + str(len(validationLabels)) + " (%.1f%%).") % (100.0 * correct / len(validationLabels)))
    print("Testing...")
    guesses = classifier.classify(testData)

    print("Writing classifier output...")
    outfile = open(MINICONTEST_PATH,'w')
    output = {}
    output['guesses'] = guesses;
    pickle.dump(output,outfile)
    outfile.close()
    print("Write successful.")

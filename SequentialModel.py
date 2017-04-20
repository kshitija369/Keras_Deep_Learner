#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 19 19:20:41 2017

@author: kshitija pansare 
"""

from keras.models import Sequential
from keras.layers import Dense
import numpy
from os import listdir, path
import sys

directory = "/Users/sdeshmukh1/Desktop/Kshitija/Keras_assignment/datasets_v1"
sys.stdout = open('/Users/sdeshmukh1/Desktop/Kshitija/Keras_assignment/accuracy_metric.txt', 'w')
# fix random seed for reproducibility
numpy.random.seed(7)
    
for filename in listdir(directory):
    try:
        if filename.startswith("."):
            continue
        print("Training model for train dataset of " + filename)
        # load dataset
        train_path = path.join(directory, filename) + "/train0.csv"
        test_path = path.join(directory, filename) + "/test0.csv"
        print(train_path)
            
        dataset = numpy.loadtxt(train_path, delimiter=",", skiprows=1)
        dataset_test = numpy.loadtxt(test_path, delimiter=",", skiprows=1)
        # split into input (X) and output (Y) variables
        X = dataset[:,0:dataset.shape[1]-1]
        Y = dataset[:,dataset.shape[1]-1]
        
        X_test = dataset_test[:,0:dataset.shape[1]-1]
        Y_test = dataset_test[:,dataset.shape[1]-1]
        
        # create model
        model = Sequential()
        model.add(Dense(12, input_dim=dataset.shape[1]-1, activation='relu'))
        model.add(Dense(8, activation='relu'))
        model.add(Dense(1, activation='sigmoid'))
        
        # Compile model
        model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        
        print("Evaluating model for test dataset of " + filename)
        # Fit the model
        model.fit(X, Y, epochs=150, batch_size=10, verbose=1)
        
        # evaluate the model
        scores = model.evaluate(X_test, Y_test, verbose=1)
        print("Evaluation \n%s: %.2f%%" % (model.metrics_names[1], scores[1]*100))
    except ValueError:
        print("Oops!  That was no valid number.  Try again... Dataset " + filename)
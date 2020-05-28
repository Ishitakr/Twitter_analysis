# -*- coding: utf-8 -*-
"""
Created on Mon May 25 21:16:24 2020

@author: ishita
"""

import numpy as np

class logistic_regression:
    def __init__(self,features, target,num_steps,learning_rate,add_intercept = False):
        self.features = features
        self.target = target
        self.num_steps = num_steps
        #self.weights = weights
        self.learning_rate = learning_rate
        self.add_intercept = add_intercept
    
    def sigmoid(self,z):
        return 1 /(1 + np.exp(-z))
    
    def log_likelihood(self,features,target,weights):
        features = self.features
        target = self.target
        #weights = self.weights
        score = np.dot(features,weights)
        ll = np.sum(target*score - np.log(1 + np.exp(score)))
        return ll
    
    def regression(self):
        if self.add_intercept:
            intercept = np.ones((self.features.shape[0],1))
            #print(intercept)
            self.features = np.hstack((intercept, self.features))
        weights = np.zeros(self.features.shape[1])
        for step in range(self.num_steps):
            scores = np.dot(self.features, weights)
            predictions = self.sigmoid(scores)
            targets = np.reshape(np.zeros(predictions.shape[1]),self.target)
            #target = np.mat(self.target)
            
            output_error_signal = np.subtract(targets,predictions)
            print(output_error_signal.shape)
            gradient = np.dot((self.features).T, (output_error_signal).T)
            weights = weights + self.learning_rate * gradient
        if step % 10000 == 0:
            print (self.log_likelihood(self.features, self.target, weights))
        
        return weights
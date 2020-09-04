# -*- coding: utf-8 -*-
"""
Created on Wed Jun  3 21:31:06 2020

@author: Rakin Shahriar
"""

import numpy as np
import tensorflow.compat.v1 as tf
from tensorflow.python.framework import ops

class AutoRec(object):
    def __init__(self, visibleDimensions, epochs =200 , hiddenDimensions =50, learningRate = 0.1, batchSize = 100):
        
        self.visibleDimensions =  visibleDimensions
        self.epochs = epochs
        self.hiddenDimensions = hiddenDimensions
        self.learningRate = learningRate
        self.batchSize = batchSize
        
    def Train(self, X):
        
        ops.reset_default_graph()
        
        self.MakeGraph()
        
        init = tf.global_variables_initializer()
        self. sess = tf.Session()
        self.sess.run(init)
        
        npX = np.array(X)
        
        for epoch in range(self.epochs):
            
            for i in range(0, npX.shape[0], self.batchSize):
                epochX = npX[i:i+self.batchSize]
                self.sess.run(self.update, feed_dict={self.inputLayer: epochX})
                
    def GetRecommendations(self, inputUser):
        rec = self.sess.run(self.outputLayer, feed_dict = {self.inputLayer: inputUser})
        return rec[0]
    
    def MakeGraph(self):
        tf.set_random_seed(0)
        
        tf.disable_eager_execution()
        
        self.encoderWeights = {'weights': tf.Variable(tf.random_normal([self.visibleDimensions, self.hiddenDimensions]))}
        self.decoderWeights = {'weights': tf.Variable(tf.random_normal([self.hiddenDimensions, self.visibleDimensions]))}
        
        self.encoderBiases = {'biases': tf.Variable(tf.random_normal([self.hiddenDimensions]))}
        self.decoderBiases = {'biases': tf.Variable(tf.random_normal([self.visibleDimensions]))}
        
        
        self.inputLayer = tf.placeholder('float', [None, self.visibleDimensions])
        
        
        hidden = tf.nn.sigmoid(tf.add(tf.matmul(self.inputLayer, self.encoderWeights['weights']), self.encoderBiases['biases']))
        
        
        self.outputLayer = tf.nn.sigmoid(tf.add(tf.matmul(hidden , self.decoderWeights['weights']), self.decoderBiases['biases']))
        
        self.labels = self.inputLayer
        
        loss = tf.losses.mean_squared_error(self.labels, self.outputLayer)
        optimizer = tf.train.RMSPropOptimizer(self.learningRate).minimize(loss)
        
        self.update = [optimizer, loss]
        
        
        
        
        
        
        
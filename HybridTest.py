# -*- coding: utf-8 -*-
"""
Created on Thu Apr 16 13:59:57 2020

@author: Rakin Shahriar
"""

from MovieLens import MovieLens
from RBMAlgorithm import RBMAlgorithm
from AutoRecAlgorithm import AutoRecAlgorithm
#from ContentKNNAlgorithm import ContentKNNAlgorithm
from HybridAlgorithm import HybridAlgorithm
from Evaluator import Evaluator
#import model
#import tensorflow as tf
#from main import gru
import random 
import numpy as np

def loadMovieLensData():
    ml = MovieLens()
    print("Loading Movie ratings...")
    data = ml.loadMovieLensLatestSmall()
    print("\nComputing movie popularity ranks so we can measure novelty later...")
    rankings = ml.getPopularityRanks()
    return (ml, data, rankings)

np.random.seed(0)
random.seed(0)

(ml, evaluationData, rankings) = loadMovieLensData()

evaluator = Evaluator(evaluationData, rankings)

SimpleRBM = RBMAlgorithm(epochs = 60)

SimpleAutoRec = AutoRecAlgorithm(epochs = 100)

Hybrid = HybridAlgorithm([SimpleRBM, SimpleAutoRec], [0.5, 0.5])

evaluator.AddAlgorithm(SimpleRBM, "RBM")
evaluator.AddAlgorithm(SimpleAutoRec, "Auto Encoder")
evaluator.AddAlgorithm(Hybrid, "Hybrid")

evaluator.Evaluate(False)
evaluator.SampleTopNRecs(ml)
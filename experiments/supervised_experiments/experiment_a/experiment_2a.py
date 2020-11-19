'''
architecture:

convolution -> pooling sub-sampling -> convolution -> pooling sub-sampling-> convolution -> fully connected -> fully connected

from scratch
'''

import keras 
from keras.models import Sequential
from keras.layers import Dense, MaxPooling2D, Conv2D, Flatten, Dropout 
from keras.optimzers import Adam 
from keras.callbacks import TensorBoard 
from keras.utils import np_utils 

import itertools

import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

from sklearn.model_selection import test_train_split 
from sklearn.metrics import confusion_matrix 
from sklearn.metrics import classification_report 
from sklearn.metrics import roc_curve, auc 
from sklearn.metrics import accuracy_score 


'''

633 total items in data

506 in train
64 in test
63 in validation

'''
class CNN_SVM():

    def __init__(self):
        '''
        Initializes the cnn-svm model
        '''
    
    def model():
        '''
        builds model base
        '''
        # specify batch size and number of features
        with tf.name_scope('input'):
            x_input = tf.placeholder(
                dtype = tf.float32, shape = [None, num_features], name = 'x_input'
            )
            y_input = tf.placeholder(
                dtype = tf.float32, shape = [None, num_classes], name = 'actual_label'
            )
        
        
    
    def train():
        '''
        trains model
        '''
    
    def weightVariable():
        '''
        returns weight matrix with arbitrary values
        '''
    
    def biasVariable():
        '''
        returns a bias matrix with 0.1 values
        '''
    
    def conv2d(features, weight):
        '''
        produces a convolution layer that filters image subregion
        '''
    
    def maxPool2x2(features):
        '''
        downsamples the image based on convolution layer
        '''
        
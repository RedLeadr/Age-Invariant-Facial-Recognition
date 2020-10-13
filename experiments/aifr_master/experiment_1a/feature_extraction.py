# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:26:38 2020

@author: Joseph Woodall
@author_email: josephlw4@gmail.com
"""

'''

After image_preprocessing, the next step is to extract features
as per the requirements. By using CNN, feature extraction and classification
are concerns of CNN itself with a single structure, and it extracts deeper 
2D features, AND it's fully adaptive and invariant to local and geometric
changes in the image. 

There are three main layers in the CNN: convolution layer, pooling layer, 
and output layer. Feed-forward structure is used to arrange these layers in the
network. Each convolution layer is followed by a pooling layer except for the 
last convolution layer, which is followed by the output layer. 

Convolution and pooling layers are 2D layers whereas the output layer is 1D. 
Every 2D layer of a CNN contains several places. A plane of a 2D layer consists
of 2D array of neurons. Feature map is the output of a plane.

AIFR-CNN contains a 7 layer architecture with 3 convolution layers and 2 pooling
layers with 2 fully-connected output layers. 


Input image -> layer 1-c1: convolution -> layer 2-p2: pooling ->
layer 3-c3: convolution -> layer 4-p4: pooling -> layer 5-c5: convolution ->
layer 6-f6: fully connected -> layer 7-f7: fully connected -> output

'''
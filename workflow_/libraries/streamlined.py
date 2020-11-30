from sklearn import metrics 
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_blobs

import tensorflow as tf 

import os
import shutil 
import bz2
import argparse 
import gdown
from imutils import paths 
import random 

import streamlit as st 

import cv2 
import dlib 
import imageio
import numpy as np
import pandas as pd

# performing metric check on dummy 
# may import private dataset if needed, marked with #

centers = [[-1, 1], [-1, -1], [1, -1]]
dataset_ = make_blobs(n_samples = 750, centers = centers, cluster_std = 0.4, random_state = 0)

'''
Load, encode, image
'''

st.header('Loading and encoding, all the images')
with st.spinner():
    #X, labels_true = tf.keras.preprocessing.image.load_img(dataset)
    #X, labels_true = tf.keras.preprocessing.image.img_to_array(dataset)
    X, labels_true = dataset_
    X = StandardScaler().fit_transform(X)
st.success('great success')
st.header('Clustering the images')
with st.spinner():
    db = DBSCAN(min_samples = 10, n_jobs = -1).fit(X) # metric = 'euclidean'
    core_samples_mask = np.zeros_like(db.labels_, dtype = bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_ 
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
    n_noise_ = list(labels).count(-1)
st.success('great success')

st.header('Metric Information!')
st.write('Estimated number of clusters: %d' % n_clusters_)
st.write('Estimated number of noise points: %d' % n_noise_)
st.write("Homogeneity: %0.3f" % metrics.homogeneity_score(labels_true, labels))
st.write("Completeness: %0.3f" % metrics.completeness_score(labels_true, labels))
st.write("V-measure: %0.3f" % metrics.v_measure_score(labels_true, labels))
st.write("Adjusted Rand Index: %0.3f"
      % metrics.adjusted_rand_score(labels_true, labels))
st.write("Adjusted Mutual Information: %0.3f"
      % metrics.adjusted_mutual_info_score(labels_true, labels))
st.write("Silhouette Coefficient: %0.3f"
      % metrics.silhouette_score(X, labels))
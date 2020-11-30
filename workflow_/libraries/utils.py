from sklearn import metrics 
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN

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

def unzip_bz2_file(zipped_file_name):
    zipfile = bz2.BZ2File(zipped_file_name)
    data = zipfile.read()
    newfilepath = output[:-4] #discard .bz2 extension
    open(newfilepath, 'wb').write(data)
 
def download_file(url):
    output = url.split("/")[-1]
    gdown.download(url, output, quiet=False)
    
if os.path.isfile('shape_predictor_5_face_landmarks.dat') != True:
    print("shape_predictor_5_face_landmarks.dat is going to be downloaded")
    url = "http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2"
    download_file(url)
    unzip_bz2_file(os.getcwd())

if os.path.isfile('dlib_face_recognition_resnet_model_v1.dat') != True:
    print("dlib_face_recognition_resnet_model_v1.dat is going to be downloaded")
    url = "http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2"
    download_file(url)
    unzip_bz2_file(os.getcwd())

pose_predictor_5_point = dlib.shape_predictor('shape_predictor_5_face_landmarks.dat')
face_detector = dlib.get_frontal_face_detector()
face_encoder = dlib.face_recognition_model_v1("dlib_face_recognition_resnet_model_v1.dat")

def load_image(img_src):
    img = imageio.imread(img_src)
    return img

def detect_face_locations(img, number_upsample = 1, model = 'cpu'):
    return face_detector(img, number_upsample)

def detect_one_face_axis(face_location):
    return(face_location.top(), face_location.right(), face_location.bottom(), face_location.left())

def detect_face_axis_dlib(dlib_face_locations):
    face_axis = {}
    face = {}
    if not isinstance(dlib_face_locations, dlib.rectangles):
        print('[INFO] send correct dlib_face_locations')
        return 0
    dlib_face_locations_length = len(dlib_face_locations)
    face_axis['tot_faces'] = dlib_face_locations_length
    for num_face, dlib_face_location in zip(range(dlib_face_locations_length), dlib_face_locations):
        face['top'] = dlib_face_location.top()
        face['left'] = dlib_face_location.left()
        face['bottom'] = dlib_face_location.bottom()
        face['right'] = dlib_face_location.right()
        face_axis['face_' + str(num_face + 1)] = face 
    return face_axis 

def detect_face_axis(dlib_face_locations, lib = 'dlib'):
    return detect_face_axis_dlib

def detect_faces(img, lib = 'dlib'):
    return detect_face_locations(img)

def detect_face_landmarks(face_image, face_locations = None, lib = 'dlib', model = 'large'):
    if face_locations is None:
        face_locations = detect_faces(face_image, lib)

    if model == 'small':
        pose_predictor = pose_predictor_5_point
    return [pose_predictor(face_image, face_location) for face_location in face_locations]

def img_to_encoding(face_image, known_face_locations = None, num_jitters = 1, lib = 'dlib'):
    raw_landmarks = detect_face_landmarks(face_image, known_face_locations, lib, model = 'small')
    return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]
    

def load_images_to_encode(face_paths):
    imagePaths = sorted(list(paths.list_images(face_paths)))
    random.seed(20)
    random.shuffle(imagePaths)
    tot_len = len(imagePaths)
    st.write('total images is {}'.format(tot_len))
    X = []
    y = []
    print('[INFO] reading face from images')
    for idx, imagePath in enumerate(imagePaths):
        if imagePath.endswith('.jpg'):
            print('{} / {}'.format(idx + 1, tot_len))
            face = img_to_encoding(imageio.imread(imagePath))
            if len(face) == 1:
                X.append(face[0])
                y.append(imagePath.split(os.path.sep)[-2])
            else:
                print('[INFO] this image, {}, did not have a face or had more than one face'.format(imagePath))
        else:
            print('[INFO] please only input .jpg files')
    unique_names = list(set(y))
    labels = [unique_names.index(name) for name in y]
    X = np.array(X)
    return (X, np.array(labels), unique_names, tot_len)
    

def load_images_to_clust(face_paths):
    imagePaths = sorted(list(paths.list_images(face_paths)))
    random.seed(20)
    random.shuffle(imagePaths)
    tot_len = len(imagePaths)
    cluster_encodings = []
    image_paths = []
    for idx, imagePath in enumerate(imagePaths):
        if imagePath.endswith('.jpg'):
            print('{} / {}'.format(idx + 1, tot_len))
            face = img_to_encoding(imageio.imread(imagePath))
            if len(face) == 1:
                cluster_encodings.append(face[0])
                image_paths.append(imagePath)
            else:
                print('[INFO] this image, {}, did not have a face or had more than one face'.format(imagePath))
        else:
            print('[INFO] please only input .jpg files')
    print('[INFO] total amount of images read = {}'.format(len(image_paths)))
    labels = np.array(cluster_encodings) # metric info labels
    labels_true = np.array(image_paths) # metric info labels_true
    return labels, labels_true

def metric_information():
    # labels, labels_true = load_images_to_clust()
    # X = load_images_to_encode()
    homogeneity = st.write('homogeneity: {}'.format(metrics.homogeneity_score(labels_true, labels)))
    completeness = st.write('completeness: {}'.format(metrics.completeness_score(labels_true, labels)))
    v_measure = st.write('v-measure: {}'.format(metrics.v_measure_score(labels_true, labels)))
    adjusted_rand = st.write('adjusted rand index: {}'.format(metrics.adjusted_rand_score(labels_true, labels)))
    adjusted_mutual = st.write('adjusted mutual info: {}'.format(metrics.adjusted_mutual_info_score(labels_true, labels)))
    silhouette = st.write('silhouette coefficient: {}'.format(metrics.silhouette_score(X, labels)))
    return homogeneity, completeness, v_measure, adjusted_rand, adjusted_mutual, silhouette



def save_faces(labels, face_paths, save_to_location = './'):
    unique_labels = list(set(labels))
    for idx, label in enumerate(labels):
        face_num = unique_labels.index(label)
        save_path = os.path.join(save_to_location, 'face {}'.format(face_num + 1))
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        shutil.copy(face_paths[idx], save_path)
    
def clustering_algorithm():
    return DBSCAN(min_samples = 1, metric = 'euclidean', n_jobs = -1, )
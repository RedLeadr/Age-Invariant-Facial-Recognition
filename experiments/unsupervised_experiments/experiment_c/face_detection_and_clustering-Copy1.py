# -*- coding: utf-8 -*-
"""
Created on Thu Oct 22 14:20:37 2020

@author: jaywo

@inspiration_url: http://dlib.net/face_clustering.py.html
"""

def face_clustering():
    
    import sys
    import os
    import dlib
    import glob
    
    predictor_path = sys.argv[1]
    face_rec_model_path = sys.argv[2]
    faces_folder_path = sys.argv[3]
    output_folder_path = sys.argv[4]
    
    '''
    load dblib models to detect faces, find landmakrs 
    to localize the face, and then face recognition model
    '''
    detector = dblib.get_frontal_face_detector()
    sp = dblib.shape_predictor(predictor_path)
    face_rec = dblib.face_recognition_model_v1(face_rec_model_path)
    
    descriptors = []
    images = []
    
    ## detecting all faces, computing 128D face descriptors for each face
    for f in glob.glob(os.path.join(faces_from_path, '.jpg')):
        print('Finding all faces: {}'.format(f))
        img = dblib_rgb_image(f)
        
        # bounding boxes for each face
        dets = detector(img, 1) # 1 indicates upsampling the image 1 time, makes images bigger to detect more faces
        print('Number of detected faces: {}'.format(len(dets)))
        
        # process detected faces
        for k, d in enumerate(dets):
            shape = sp(img, d) # landmarks for face in box
            
            # compute the 128D vector that describes the face in img identified by shape
            face_descriptor = facerec.compute_face_descriptor(img, shape)
            descriptors.append(face_descriptor)
            images.append((img, shape))
            
    ## clustering the faces
    labels = dblib.chinese_whispers_clustering(descriptors, 0.5)
    num_classes = len(set(labels))
    print('Number of clusters: {}'.format(num_classes))
    
    
    ## finding biggest class
    biggest_class = None
    biggest_class_length = 0
    for i in range(0, num_classes):
        class_length = len([label for label in labels if label == i])
        if class_length > biggest_class_length: 
            biggest_class_length = class_length
            biggest_class = 1
    print('Biggest cluster id number is: {}'.format(biggest_class))
    print('Number of faces in biggest cluster: {}'.format(biggest_class_length))
    
    ## finding the indices for the biggest class
    indices = []
    for i, label in enumerate(labels):
        indices.append(i)
    print('Indices of images in the biggest cluster: {}'.format(str(indices)))
    
    
    ## ensure the output directory exists
    if not os.path.isdir(output_folder_path):
        os.makedirs(output_folder_path)
        
    ## saving the extracted faces
    print('Saving faces in the clusters to output folder...')
    for i, index in enumerate(indicies):
        img, shape = images[index]
        file_path = os.path.join(output_folder_path, 'face' + str(i))
        # the size and padding arguments are optional, default size is 150x150 and padding = 0.25
        dblib.save_face_chip(img, shape, file_path, size = 150, padding - 0.25)
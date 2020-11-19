'''
AIFR by coupled auto-encoder network

http://chenfeixu.com/wp-content/uploads/2014/04/CAN.pdf


See experiments/experiment_d/README.md for details
'''

import os
import glob
import pickle 
import numpy as np
from PIL import Image 
import cv2 as cv
import sys
import dlib
from sklearn import metrics 


from experiment_d import settings, main, utils

class data():

    def __init__(self, transform = None, istrain = False, isvalid = False, isquery = False, isgall1 = False, isgall2 = False, isgall3 = False):
        super().__init__()
        self.metafile = os.path.join('/path/to/dataset')

        with open(self.metafile, 'rb') as fd: 
            
            
            predictor_path = sys.argv[1]
            faces_folder_path = sys.argv[3]
            output_folder_path = sys.argv[4]
            
            '''
            load dblib models to detect faces, find landmakrs to localize the face
            '''
            detector = dblib.get_frontal_face_detector()
            sp = dblib.shape_predictor(predictor_path)
            
            descriptors = []
            images = []
            
            ## detecting all faces, computing 128D face descriptors for each face
            for f in glob.glob(os.path.join(faces_folder_path, '.jpg')):
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

        ''' '''
        self.labels = list(self.morph_dict.keys())
        self.images = list(self.morph_dict.values())
        self.root_path = settings.args.root_path 

        self.transform = transform 
        self.istrain = istrain 
        self.isvalid = isvalid 
        self.isquery = isquery 
        self.isgall1 = isgall1
        self.isgall2 = isgall2 
        self.isgall3 = isgall3 
        
        self.list_test_ids = []
        self.list_test_ages = []
        self.all_test_list = []
        self.test_query_list = []
        self.test_gall1_list = []
        self.test_gall2_list = []
        self.test_gall3_list = []
        
        self.list_valid_ids = []
        self.list_valid_ages = []
        self.all_valid_list = []
        self.valid_query_list = []
        self.valid_gall1_list = []
        self.valid_gall2_list = []
        self.valid_gall3_list = []

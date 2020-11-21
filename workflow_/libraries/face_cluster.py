import os 
import streamlit as st
from utils import load_images_to_clust, save_faces, clustering_algorithm

class faceCluster():

    def __init__(self, dataset):
        self.valid_path = False 
        if os.path.isdir(dataset):
            self.valid_path = True
            self.path = dataset 
        else:
            print('[ERROR] please set a valid dataset path')
        
    def load_faces(self):
        if self.valid_path:
            self.encs, self.paths = load_images_to_clust(self.path)
            self.model = clustering_algorithm()
            self.model.fit(self.encs)
            self.tot_faces_list = self.model.labels_
        else:
            print('[ERROR] please set a valid dataset path')
    
    def save_faces(self, save_location = None):
        if save_location is not None:
            save_faces(self.tot_faces_list, self.paths, save_location)
        else:
            save_faces(self.tot_faces_list, self.paths)


        
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:24:52 2020

@author: josephwoodall
"""

'''
Dataset @: /Users/josephwoodall/ds_personal_projects/brooks_family_photo_project/brooks_family_photo_project_original_dataset/jpeg

---

Extract Transform Load
'''

Class ETL:
    def convert_from_input():
    
    '''
    
    Currently only supports pdf files as inputted format.
    
    Flow: 
    pdf file input -> convert to jpg -> load jpg files to NumPy Array
    
    '''
        import time
        then = time.time()
        import os
        from pdf2image import convert_from_path
        from workflow_.libraries.data import data

        os.system('brew install poppler') # install on requirements txt
        os.system('pip instlal pdf2image') # install install on requirements txt

        # directory = os.chdir('/Users/josephwoodall/brooks_family_photo_project/brooks_family_photo_project_original_dataset') # directory containing original dataset
        # path = ('/Users/josephwoodall/brooks_family_photo_project/brooks_family_photo_project_original_dataset/jpeg') # directory containing jpeg of original dataset
    
        # Iteratively convert images in directory to .jpg
        for filename in os.listdir(directory):
            if filename.endswith('.pdf'): # checks folder for .pdf images, we may want to allow for multiple formats for versatility
                convert_from_path(filename, output_folder = path, fmt = 'jpeg')
            else:
                return "Please input all files as pdf format"
    
        # Iteratively converts .jpg files to NumPy arrays
        for filename in os.listdir(path):
            if filename.endswith('.jpg'):
                image.imread()
            else:
                continue

        now = time.time()
        print('It took: ', now - then, " seconds to run this script")
    
    def main():
        if __name__ == '__main__':
            main()

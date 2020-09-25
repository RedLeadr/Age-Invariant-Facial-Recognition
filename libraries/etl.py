#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:24:52 2020

@author: josephwoodall
"""

'''
Dataset @: /Users/josephwoodall/Downloads/Brooks_original_photo_dataset

---

Extract Transform Load
'''

def ETL():
    import os
    from pdf2image import convert_from_path

    os.system('brew install poppler')
    os.system('pip instlal pdf2image')

    directory = os.chdir('/Users/josephwoodall/Downloads/Brooks_original_photo_dataset')
    path = ('/Users/josephwoodall/Downloads/Brooks_original_photo_dataset/jpeg')
    
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'): # checks folder for .pdf images, we may want to allow for multiple formats for versatility
            convert_from_path(filename, output_folder = path, fmt = 'jpeg')
        else:
            return "Please input pdf format"
    directory = os.chdir('/Users/josephwoodall/Downloads/Brooks_original_photo_dataset/jpeg')
    list = os.listdir(directory)
    number_files = len(list)
    return "The number of well changed pdf images into target folder is: " + str(number_files)
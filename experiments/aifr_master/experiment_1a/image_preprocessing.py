# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 13:25:03 2020

@author: Joseph Woodall
@author_email: josephlw4@gmail.com
"""

'''

Image preprocessing helps keep the dataset in normalized format. 

It includes detection and cropping of facial portion from the given image. 

For this purpose, we will use the Viola Jones algorithm for facial detection. 

The next step is to conver the RGB image to gray scale image. 

Later, these images are resized to 32x32 pixels.

'''

## Importing the libraries and data
import cv2 as cv

## Open and iterate over multiple images in dataset
directory = 'Users/jaywo/Downloads/Brooks_original_photo_dataset/jpeg/*.jpg'

original_image = cv.imread(directory) # rbg images
grayscale_image = cv.cvtColor(original_image, cv.COLOR_BGR2GRAY) # converting rgb image to grayscale

# Loading the viola jones classifier and create a cascade object for face detection
face_cascade = cv.CascadeClassifier('haarcascade_frontalface_alt.xml') # be sure to have this file in /experiment1a
detected_faces = face_cascade.detectMultiScale(grayscale_image)

## Resizing images to 32x32 pixels
cv.resize(directory, (32, 32))
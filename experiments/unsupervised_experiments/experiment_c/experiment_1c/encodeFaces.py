import os 
import time
import cv2 
import pickle
import argparse
import face_recognition # to write own face_recognition library for customization
from imutils import paths 

from settings import encodings_path

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--dataset', required = True, help = 'path to the input directory of faces and images')
ap.add_argument('-e', '--encodings', required = True, help = 'path to serialized database of facial encodings')
ap.add_argument('-d', '--detection_method', type = str, default = 'cnn', help = 'face detection model to use: either hog or cnn')
args = vars(ap.parse_args())

start_time = time.time()

print('quantifying faces...')
imagePaths = list(paths.list_images(args['dataset']))
data = []

for (i, imagePath) in enumerate(imagePaths):
    print('processing image {}/{}'.format(i + 1, len(imagePaths)))
    print(imagePath)

    image = cv2.imread(imagePath)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    boxes = face_recognition.face_locations(image, model = args['detection_method'])
    encodings = face_recognition.face_encodings(image, boxes)

    d = [{'imagePath' : imagePath, 'loc' : box, 'encoding' : enc}
            for (box, enc) in zip(boxes, encodings)]
    data.extend(d)

print('serializing encodings...')
f = open(args['encodings'], 'wb')
f.write(pickle.dumps(data))
f.close()
print('encodings of images saved in {}'.format(encodings_path))
elapsed_time = time.time() - start_time
print('total time elapsed {} seconds'.format(elapsed_time))
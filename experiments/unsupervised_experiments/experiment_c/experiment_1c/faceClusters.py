import numpy as np 
from sklearn.cluster import DBSCAN 
import argparse 
import pickle 
import cv2 
import shutil 
import os 

from settings import face_data_path, encodings_path, clustering_result_path

def moveImage(image, id, labelID):

    path = clustering_result_path + '/label' + str(labelID)
    if os.path.exists(path) == False:
        os.mkdir(path)
    
    filename = str(id) + '.jpg'

    cv2.imwrite(os.path.join(path, filename), image)

    return

ap = argparse.ArgumentParser()
ap.add_argument('-e', '--encodings', required = False, help = 'path to serialized db of facial encodings')
ap.add_argument('-j', '--jobs', type = int, default = -1, help = 'number of parallel jobs to run (-1 will use all CPUs)')
args = vars(ap.parse_args())

'''Load the serialized face encodings + bounding box locations from disk/encodings pickle file, then extract the set of encodings so we can cluster them'''
print('loading encodings...')
data = pickle.loads(open(args['encodings'], 'rb').read())
data = np.array(data)
encodings = [d['encoding'] for d in data]

'''Cluster the embeddings'''
print('clustering...')
clt = DBSCAN(metric = 'euclidean', n_jobs = args['jobs'])
clt.fit(encodings)
labelIDs = np.unique(clt.labels_) # number of unique faces found in dataset
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print('number of unique faces: {}'.format(numUniqueFaces))

for labelID in labelIDs: # loop over unique face integers
    print('faces for face id: {}'.format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size = min(25, len(idxs)), replace = False)

    faces = [] # initialize list of faces for montage

    for i idxs: # loop over sampled indexes
        image = cv2.imread(data[i]['imagePath'])
        (top, right, bottom, left) = data[i]['loc']
        face = image[top:bottom, left:right]

        move_image(image, i, labelID) # moving image to cluster folder

        face = cv2.resize(face(96, 96))
        faces.append(face)
    
    montage = build_montages(faces, (96, 96), (5, 5))[0]

    title = 'Face ID # {}'.format(labelID)
    title = 'Unknown Faces' if labelID == -1 else title 

    cv2.imwrite(os.path.join(clustering_result_path, title + '.jpg'), montage)
    
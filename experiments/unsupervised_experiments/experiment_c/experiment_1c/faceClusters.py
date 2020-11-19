import numpy as np 
from sklearn.cluster import DBSCAN 
from sklearn import metrics 
# sklearn preprocessing
import matplotlib.pyplot as plt 
import argparse 
import pickle 
import cv2 
import shutil 
import os 
import time 
from imutils import build_montages

from settings import face_data_path, encodings_path, clustering_results_path

start_time = time.time()

def move_image(image, id, labelID):

    path = clustering_results_path + '/label' + str(labelID)
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
clt = DBSCAN(eps = 0.5, metric = 'euclidean', n_jobs = args['jobs']).fit(encodings)
labelIDs = np.unique(clt.labels_) # number of unique faces found in dataset
X = clt.labels_
numUniqueFaces = len(np.where(labelIDs > -1)[0])
print('number of unique faces: {}'.format(numUniqueFaces))
n_clusters = len(set(X)) - (1 if 1 in X else 0)
n_noise = list(X).count(1)

for labelID in labelIDs: # loop over unique face integers
    print('faces for face id: {}'.format(labelID))
    idxs = np.where(clt.labels_ == labelID)[0]
    idxs = np.random.choice(idxs, size = min(25, len(idxs)), replace = False)

    faces = [] # initialize list of faces for montage

    for i in idxs: # loop over sampled indexes
        image = cv2.imread(data[i]['imagePath'])
        (top, right, bottom, left) = data[i]['loc']
        face = image[top:bottom, left:right]

        move_image(image, i, labelID) # moving image to cluster folder

        face = cv2.resize(face, dsize = (96, 96))
        faces.append(face)
    
    '''Visualizing clusters'''

    '''Montages'''
    montage = build_montages(faces, (96, 96), (5, 5))[0]

    title = 'Face ID # {}'.format(labelID)
    title = 'Unknown Faces' if labelID == -1 else title 

    cv2.imwrite(os.path.join(clustering_results_path, title + '.jpg'), montage)



'''Metric Information'''
print('#### METRIC INFORMATION ####')
print('Number of clusters: %d' % n_clusters)
print('Number of noise points: %d' % n_noise)
# print('Silhouette Coefficient: %0.3f' % metrics.silhouette_score(X, X))
elapsed_time = time.time() - start_time
print('total time elapsed {} seconds'.format(elapsed_time))
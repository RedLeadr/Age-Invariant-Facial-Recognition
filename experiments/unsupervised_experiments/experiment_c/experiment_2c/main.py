import sys
import os
import dlib
import glob

if len(sys.argv) != 3:
    print("Please specify valid arguments. Call the program like this \npython face_clustering.py -specify input folder- -specify output path-")
    exit()
predictor_path = '/Users/jaywo/Downloads/shape_predictor_68_face_landmarks.dat.bz2'
face_rec_model_path = '/Users/jaywo/Downloads/dlib_face_recognition_resnet_model_v1.dat.bz2' 

faces_folder_path = sys.argv[1] # '/Users/jaywo/ds_personal_projects/brooks_family_photo_project/dataset'

output_folder_path = sys.argv[2] # '/Users/jaywo/ds_personal_projects/brooks_family_photo_project/experiments/unsupervised_experiments/experiment_c/experiment_2c'

min_detected_faces = 633

'''
load dlib models to detect faces, find landmakrs 
to localize the face, and then face recognition model
'''
detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor(predictor_path)
face_rec = dlib.face_recognition_model_v1(face_rec_model_path)

descriptors = []
images = []

## detecting all faces, computing 128D face descriptors for each face
for f in glob.glob(os.path.join(faces_folder_path, '.pdf')):
    print('Finding all faces: {}'.format(f))
    img = dlib_rgb_image(f)
    
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
'''
label options:
dlib.bottom_up_cluster_clustering()
dlib.chinese_whispers_clustering()
dlib.modularity_clustering()
dlib.spectral_cluster_clustering()
'''
labels = dlib.chinese_whispers_clustering(descriptors, 0.5)
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
for i, index in enumerate(indices):
    img, shape = images[index]
    file_path = os.path.join(output_folder_path, 'face' + str(i))
    # the size and padding arguments are optional, default size is 150x150 and padding = 0.25
    dlib.save_face_chip(img, shape, file_path, size = 150, padding = 0.25)

'''
class test():
    def detectionTest():
        
        total_output = len([name for name in outupt_folder_path])
        if total_output.min() == min_detected_faces:
            print('at least 633 faces found')
        else: 
            print('Captain, there is a running amuck, we dont have all the faces')
    
    def recognitionTest():
        
        # take test and validation from original output_folder_path

'''

'''
Two tests: detection and recognition

Detection:
Find the max faces during a test, set that as the ceiling; compare

Recogntion: 
training, testing, and validation sets (70, 15, 15) or (80, 10, 10)
'''

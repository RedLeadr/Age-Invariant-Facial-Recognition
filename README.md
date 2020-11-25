This code is the version-1-tool that stemmed from the question, "how do I help my family recognize who is in the hundreds of family photos we have?" Through my knowledge of deep learning, unsupervised learning, and tensors, I achieved a unique age-invariant model. I used novel combinations of pre-processing techniques to build something that solves a generally hard issue of finding faces of family members across their lifespan. You may check the experiments for method selection under the experiments directory. 

How to run: 

    Put original .jpg image files into the workflow_/libraries/original_dataset folder
    Navigate to workflow_/libraries directory and run $ streamlit run main.py
    Clustered images will be in the /output folder, separated by each unique face

Encoding Algorithm: 

    Images are mapped to a two dimensional x, y coordinate plane capturing each image's facial landmarks, measuring a variety of metrics such as the face's width of nose, depth 
    of eye sockets, distance between each facial landmark (see dlib for exact landmarks); this two dimensional array is a histogram of each image's pixels. The two dimensional
    facial landmarks correspond to the location on the original image on an x, y plane. Density based spatial application with noise (DBSCAN) uses this array as input. 
    The core points are derived by the conversion of each image's histogram of pixels and mapping them to the 128 dimensional face descriptor. This face descriptor, an array of arrays, 
    is calculated by taking the average number of iterations of landmark calculation by the number of times the image is randomly jittered during the process.

Clustering Algorithm:
    
    Density based spatial applications with noise (DBSCAN) operates by defining a cluster with a max set of connected points, which in this instance, are the array of arrays of the histogram of image pixels durived during the encoding algorithm. It has two parameters: epsilon and minimum points:
        Epsilon is the maximum radius of the neighborhood around a core point. 
        Minimum points is the minimum number of points in the epsilon neighborhood to define a cluster. (note, core points and points are mutually exclusive)
    There are three classifications of these points: core; border; and outlier:
        A core point has at least minimum points in its epsilon neighborhood at the interior of a cluster.
        A border point has less than the minimum points but can still be reached by a cluster (of one core point).
        An outlier, also known as a noise point, is a point that cannot be reached by a cluster.
    For example: a point, y, is 'reachable' from point x if there is a path P1(which is point x) and Pn(which is point y), where each Pi+1 on the path must be core paths with the possible
    exception of Pn. An object is directly density-reachable from point x, if x is a core object and y is in x's epsilon neighborhood. 
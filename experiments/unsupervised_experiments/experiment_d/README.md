
AIFR by coupled auto-encoder network

http://chenfeixu.com/wp-content/uploads/2014/04/CAN.pdf

Pytorch implementation



CAN: coupled auto-encoder network
    - identical auto-encoders 
    - two single-hidden-layer neural networks as a bi-directional bridge
    - age-invariant features I(1) and I(2) derived by basic reconstruction and transer
        - basic reconstruction (1st step) reconstructs facial image inputs
        x1 and x2 independently by two auto-encoders to capture
        as much as main factors of input; inputs are projected
        into a high-dimensional feature space in hidden layers
            - two basic blocks, encoder and decoder
        - transfer (2nd step) imposes constraints in the above feature space
        to nonlinearly decompose it into three feature subspaces:
        identity feature space which is age-invariant, age feature space 
        which is identity-independent and a noise space
            - impose constraints in hiden layer to decompose it into 
            three subspaces
    - training
        - set training set to 1; initialize identity-realted and 
        age-related parameters and noise-related parameters to 0
        - repeat
        - shuffle T
        - repeat 
            - pick a mini-batch T' from T without overlapping
            - compute min cost function
            - update parameters(1) by solving min cost function
            - compute min cost function
            - update parameters(2) by solving min cost function
        - until T is looped over
        - t = t + 1
        - until maxEpoch is met
    - training is unsupervised, extracted age-invariant features 
    I1 and I2 in hidden layers are note discriminative, so 
    they cant be directly used for face recognition and retrieval
        - PCA on extracted I1 and I2 followed by LDA to make I1 and I2
        more compressed and discriminative as the final age-invariant features
        for face recognition and retrieval
    
Datasets:
    - three public aging face datasets: 
        - fgnet; 1002 images of 82 different people, each one has about
         13 images on average and age range from 0 to 69
        - cacd; 163,466 face images of 2000 people with age range of 16 to 62
        - cacd-vs; subset of cacd, 2000 positive pairs and 2000 negative pairs
    - private dataset:
        - family photo dataset images with at least 633 images
    - face preprocess
        - convert faces to gray ones if RGB images
       - detect locations of faces in images
       - locate the 83 landmarks using Face ++ API
        - align the images to make eyes located at the same horizontal positions
        - crop images to remove the background and hair region
        - rescale them by bicubic interpolation 
        - reshape them into one-dimensional vector
        - map data into [0, 1]
        - normalize to have zero mean

Parameters setting 
    - hyperparameters
        - input dimension n
        - identity feature dimension p
        - age feature dimension q
        - noise-related feature dimension r
        - the number of bridge neurons k
        - dimension of PCA and LDA
    - presets: 
        - n = 35x32
        - p = 2100
        - q = 600
        - r = 300
        - k = 500
        - PCA dimension reduction = 400
        - LDA dimension reduction = 100
        - learning rate a = 0.0001
        - mini-batch size m = 10 to perform SGD
        - maxEpoch = 100, 500, 800
        - momentum = 0.9
    - Evaluation metrics
        - strategy: leave-one-image-out with rank-k identification
        rates, one image as test sample and train model using remaining images; 
        repeat process 1002 times and take average as final identification rates; 
        cosine similarity to compute matching scores between test example and remaining images
        - for rank-k, sort the matching results from top-1 to top-k for each test example; 
        then get rank-k identification rates after averaging results

Expected output (on fgnet)
    - performance:
        - 92.3 percent accurate on verification (mean average precision)
    - comparison:
        - LF-CNNs = 98.5 percent accurate on verification (mean average precision)
    - "upgrade" to LF-CNN if needed (https://arxiv.org/pdf/1809.00338.pdf)

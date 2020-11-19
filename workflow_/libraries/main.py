import face_cluster, utils, preprocessing

from sklearn import metrics 

import streamlit as st
import time 
import os 
import glob 

time.sleep(3)
# Hello There gif
# https://media.giphy.com/media/Nx0rz3jtxtEre/giphy.gif
st.markdown(
    '![Alt Text](https://media.giphy.com/media/Nx0rz3jtxtEre/giphy.gif)'
)

time.sleep(2.5)

st.title('Let us identify some faces, shall we?')

time.sleep(2.5)

with st.spinner('Doing a little prep work right now...please wait'):
    # https://media.giphy.com/media/lJNoBCvQYp7nq/giphy.gif
    st.markdown(
        '![Alt Text](https://media.giphy.com/media/lJNoBCvQYp7nq/giphy.gif)'
    ) 
    preprocessing.detect_crop() ## draw a progress bar 

time.sleep(3)


with st.spinner(''):

    st.header('Everything is good to go! Running the code now')
    # Patrick gif
    # https://media.giphy.com/media/l41Ym49ppcDP6iY3C/giphy.gif
    st.markdown(
        '![Alt Text](https://media.giphy.com/media/l41Ym49ppcDP6iY3C/giphy.gif)'
    )
    model = face_cluster.faceCluster('./dataset')
    st.write('Loading the faces from images...please wait')
    start = time.time()
    model.load_faces()
    st.write('Saving those faces into their folders...almost finished')
    model.save_faces('./output') ## display progress bar
    executionTime = (time.time() - start) / 60

st.balloons()
st.header('Ding! All Done!')

time.sleep(2.5)

st.write('Found {} unique faces from the images!'.format(len(os.path.join('output'))))



st.header('Here are the metrics!')
st.write('This script took {} minutes to run!'.format(executionTime))
metrics = utils.metric_information()
st.write(metrics)
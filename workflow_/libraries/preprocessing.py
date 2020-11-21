import dlib 
import os
import matplotlib.pyplot as plt
import streamlit as st

from PIL import Image 

def detect_crop():

    if len(os.listdir('/Users/jaywo/ds_personal_projects/brooks_family_photo_project/workflow_/libraries/dataset')) == 0:

        # dirname is directory relative to script where files to detect and crop
        dirname = '/Users/jaywo/ds_personal_projects/brooks_family_photo_project/workflow_/libraries/original_dataset'
        # put_dirname is directory where cropped images will be written 
        put_dirname = '/Users/jaywo/ds_personal_projects/brooks_family_photo_project/workflow_/libraries/dataset'
        # the width and heigh in pixels of saved images, change as needed
        crop_width = 108 
        # face crop(true) or to include other elements (false)
        simple_crop = True

        face_detector = dlib.get_frontal_face_detector()


        file_types = ('.jpg', '.jpeg', '.JGP', '.JPEG', '.png', '.PNG')

        files = [file_i
                for file_i in os.listdir(dirname)
                if file_i.endswith(file_types)]

        filenames = [os.path.join(dirname, fname)
                    for fname in files]

        # face detection on the image(s)
        print('found %d files' %len(filenames))
        
        filename_inc = 0

        filecount = 1
        with st.spinner('Finding some faces from these pictures'):
            for file in filenames:
                img = plt.imread(file)
                detected_faces = face_detector(img, 1)
                print('[%d of %d] %d detected faces in %s' % (filecount, len(filenames), len(detected_faces), file))            
                for i, face_rect in enumerate(detected_faces):
                    width = face_rect.right() - face_rect.left()
                    height = face_rect.bottom() - face_rect.top()
                    if width >= crop_width and height >= crop_width:
                        image_to_crop = Image.open(file)

                        if simple_crop: 
                            crop_area = (face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom())
                        else:
                            size_array = []
                            size_array.append(face_rect.top())
                            size_array.append(image_to_crop.height - face_rect.bottom())
                            size_array.append(face_rect.left())
                            size_array.append(image_to_crop.width - face_rect.right())
                            size_array.sort()
                            short_side = size_array[0]
                            crop_area = (face_rect.left() - size_array[0], face_rect.top() - size_array[0], face_rect.right() + size_array[0], face_rect.bottom() + size_array[0])

                        cropped_image = image_to_crop.crop(crop_area)
                        crop_size = (crop_width, crop_width)
                        cropped_image.thumbnail(crop_size)
                        cropped_image.save(put_dirname + '/' + str(filename_inc) + '.jpg')
                        filename_inc += 1
                    filecount += 1
        st.success('Found the faces!')
    else:
        st.subheader('Looks like you already did your homework! Onward!')
brooks_family_photo_project/experiments

---PURPOSE---
experiment with new features, datasets, etl, and model building before putting in production folder /workflow_

---PROBLEM ONE---

In order to circumnavigate the age distribution problem:

we need a dataset that includes labeled faces with ages, with the same individual with multiple ages (~0 - ~70)

We may need to create such a dataset

Existing Datasets:
UTKFace: https://susanqq.github.io/UTKFace/ | https://www.kaggle.com/jangedoo/utkface-new ("may be used for face detection, age estimation, age progression/regression, landmark localization, etc.")

Creating datasets:
Age progression/regression by conditional adversarial autoencoder: https://github.com/ZZUTK/Face-Aging-CAAE | http://web.eecs.utk.edu/~zzhang61/docs/papers/2017_CVPR_Age.pdf


We also may only need to use Age-Invariant Facial Recognition:
https://www.researchgate.net/publication/326403604_Age_Invariant_Face_Recognition_using_Convolutional_Neural_Network

Example code of AIFR:
https://paperswithcode.com/task/age-invariant-face-recognition/codeless
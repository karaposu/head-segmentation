sample_preprocessed_img="/home/enes/lab/preprocessed_dset_dir/images/0.jpg"
sample_preprocessed_segmap="/home/enes/lab/preprocessed_dset_dir/segmaps/0.png"

import cv2

img=cv2.imread(sample_preprocessed_img, cv2.IMREAD_UNCHANGED )
print(img.shape)

img=cv2.imread(sample_preprocessed_segmap, cv2.IMREAD_UNCHANGED )
print(img.shape)


import numpy as np

# x = np.array([1,1,1,2,2,2,5,25,1,1])
unique, counts = np.unique(img, return_counts=True)
print(np.asarray((unique, counts)).T)

anno_sample="/home/enes/lab/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_mouth.png"

img=cv2.imread(anno_sample, cv2.IMREAD_UNCHANGED )
print(img.shape)

unique, counts = np.unique(img, return_counts=True)
print(np.asarray((unique, counts)).T)

anno_sample2="/home/enes/lab/CelebAMask-HQ/CelebAMask-HQ-mask-anno/0/00000_r_eye.png"

img=cv2.imread(anno_sample2, cv2.IMREAD_UNCHANGED )
print(img.shape)

unique, counts = np.unique(img, return_counts=True)
print(np.asarray((unique, counts)).T)
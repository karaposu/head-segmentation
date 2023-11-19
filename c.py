import os
# def get_latest_model_checkpoint_path(parent_dir):
#     file_name = 'last.ckpt'
#     folders = os.listdir(parent_dir)
#
#     folders_with_file = [folder for folder in folders if os.path.isfile(os.path.join(parent_dir, folder, file_name))]
#     sorted_folders = sorted(folders_with_file, reverse=True)
#     print(folders)
#     latest_folder = sorted_folders[0] if sorted_folders else None
#
#     return latest_folder

import os


def get_latest_model_checkpoint_path(parent_dir):
    file_name = 'models/last.ckpt'
    date_folders = os.listdir(parent_dir)
    latest_checkpoint_path = None

    for date_folder in date_folders:
        time_folders = os.listdir(os.path.join(parent_dir, date_folder))
        time_folders_with_file = [time_folder for time_folder in time_folders if
                                  os.path.isfile(os.path.join(parent_dir, date_folder, time_folder, file_name))]
        sorted_time_folders_with_file = sorted(time_folders_with_file, reverse=True)
        if sorted_time_folders_with_file:
            latest_checkpoint_path = os.path.join(parent_dir, date_folder, sorted_time_folders_with_file[0])
            break

    return latest_checkpoint_path

#
# def get_latest_model_checkpoint_path(parent_dir):
#     file_name = 'models/last.ckpt'
#     date_folders = os.listdir(parent_dir)
#     latest_folder = None
#
#     for date_folder in date_folders:
#         time_folders = os.listdir(os.path.join(parent_dir, date_folder))
#         time_folders_with_file = [time_folder for time_folder in time_folders if os.path.isfile(os.path.join(parent_dir, date_folder, time_folder, file_name))]
#         sorted_time_folders_with_file = sorted(time_folders_with_file, reverse=True)
#         if sorted_time_folders_with_file:
#             latest_folder = sorted_time_folders_with_file[0]
#             break
#
#     return latest_folder

# parent_dir="/home/enes/lab/head-segmentation/training_runs"
# a=get_latest_model_checkpoint_path(parent_dir)
# print(a)

import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

pp="/home/enes/lab/preprocessed_dset_dir/segmaps/0.png"
# img=cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
img=cv2.imread(pp)
print(np.unique(img))
print(img.shape)

mask0 = img[:, :, 0]
mask1 = img[:, :, 1]
mask2 = img[:, :, 2]

print("mask0 unique:", np.unique(mask0))
print("mask1 unique:", np.unique(mask1))
print("mask2 unique:", np.unique(mask2))



# plt.figure(figsize=(80,60))
plt.imshow(mask0, cmap="gray")
plt.savefig('./mask0.png')

plt.imshow(mask1, cmap="gray")
plt.savefig('./mask1.png')

plt.imshow(mask2, cmap="gray")
plt.savefig('./mask2.png')

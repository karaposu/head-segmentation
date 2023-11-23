import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_segmaps(segmap_sample_path):

    # img=cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(segmap_sample_path)

    print("        segmap shape:", img.shape)

    unique, counts = np.unique(img, return_counts=True)
    print("        segmap unique values and counts:", )
    print(np.asarray((unique, counts)).T)



if __name__ == "__main__":
    segmap_sample_path = "/home/enes/lab/preprocessed_dset_dir/segmaps/0.png"
    check_segmaps(segmap_sample_path)

    "/home/enes/lab/CelebAMask-HQ/CelebA-HQ-img/27989.png"

    # /home/enes/lab/CelebAMask-HQ/CelebA-HQ-img

    images_path= "/home/enes/lab/preprocessed_dset_dir/images/27989.png"


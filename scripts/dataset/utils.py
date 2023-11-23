import cv2
import numpy as np
import matplotlib.pyplot as plt

def check_segmaps(segmap_sample_path):

    # img=cv2.imread(pp, cv2.IMREAD_GRAYSCALE)
    img = cv2.imread(segmap_sample_path)

    print("        segmap shape:", img.shape)
    print("        segmap unique values:",np.unique(img))


    mask0 = img[:, :, 0]
    mask1 = img[:, :, 1]
    mask2 = img[:, :, 2]

    print("         mask0 unique:", np.unique(mask0))
    print("         mask1 unique:", np.unique(mask1))
    print("         mask2 unique:", np.unique(mask2))

    plt.imshow(mask0, cmap="gray")
    plt.savefig('./scripts/dataset/segmask0_channel0.png')

    plt.imshow(mask1, cmap="gray")
    plt.savefig('./scripts/dataset/segmask0_channel1.png')

    plt.imshow(mask2, cmap="gray")
    plt.savefig('./scripts/dataset/segmask0_channel2.png')


if __name__ == "__main__":
    segmap_sample_path = "/home/enes/lab/preprocessed_dset_dir/segmaps/0.png"
    check_segmaps(segmap_sample_path)

    "/home/enes/lab/CelebAMask-HQ/CelebA-HQ-img/27989.png"

    # /home/enes/lab/CelebAMask-HQ/CelebA-HQ-img

    images_path= "/home/enes/lab/preprocessed_dset_dir/images/27989.png"


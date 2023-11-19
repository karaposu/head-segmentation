import argparse
import logging
import shutil
import typing as t
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import hydra
from typing import Dict, List

# def parse_args() -> argparse.Namespace:
#     parser = argparse.ArgumentParser(description="Preprocess dataset.")
#
#     # fmt: off
#     parser.add_argument("--raw_dset_dir", "-r", type=Path, required=True, help="Raw CelebA dataset directory.")
#     parser.add_argument("--output_dset_dir", "-o", type=Path, help="Preprocessed CelebA dataset directory.")
#     # fmt: on
#
#     return parser.parse_args()


def create_structure(root_path: Path) -> t.Tuple[Path, Path]:
    logging.info(f"Creating dataset structure with root dir {root_path}...")
    if  isinstance(root_path, str):
        root_path = Path(root_path)

    images_path = root_path / "images"
    if not os.path.isdir(images_path):
        images_path.mkdir(parents=True, exist_ok=False)

    segmaps_path = root_path / "segmaps"
    if not os.path.isdir(segmaps_path):
         segmaps_path.mkdir(parents=True, exist_ok=False)

    return images_path, segmaps_path


def create_metadata_csv(attribute_txt: Path, output_dset_path: Path) -> None:
    logging.info(
        f"Creating metadata.csv file and saving in directory {output_dset_path}..."
    )
    with open(attribute_txt, "r") as file:
        lines = file.readlines()

    # Second lines contains headers without Filename
    headers = lines[1].split()
    headers.insert(0, "Filename")

    samples = [sample.split() for sample in lines[2:]]

    attribs_df = pd.DataFrame(samples, columns=headers)
    attribs_df = attribs_df.map(lambda x: x if x != "-1" else 0)

    metadata_csv = output_dset_path / "metadata.csv"
    if not os.path.isfile(metadata_csv):
         attribs_df.to_csv(str(metadata_csv), index=False)


def copy_jpg_images(src_path: Path, dst_path: Path) -> None:
    logging.info(f"Copying JPG images from {src_path} to {dst_path}...")
    for image_file in src_path.glob("*.jpg"):
        if not os.path.isfile(dst_path / image_file.name):
             shutil.copy(image_file, dst_path / image_file.name)


def load_mask_files(src_path: Path) -> t.Dict[str, t.List[Path]]:
    logging.info("Loading mask files...")
    head_parts =[
     'cloth', 'ear_r', 'eye_g', 'hair','hat', 'l_brow','l_ear', 'l_eye', 'l_lip', 'mouth', 'neck', 'neck_l', 'nose', 'r_brow', 'r_ear',  'r_eye',  'skin', 'u_lip']

    # head_parts = {
    #     "ear_r",
    #     "eye_g",
    #     "hair",
    #     "hat",
    #     "l_brow",
    #     "l_ear",
    #     "l_eye",
    #     "l_lip",
    #     "mouth",
    #     "nose",
    #     "r_brow",
    #     "r_ear",
    #     "r_eye",
    #     "skin",
    #     "u_lip",
    # }

    mask_dict = {}
    mask_files = sorted(list(src_path.rglob("*.png")))
    for mask_file in mask_files:
        image_filename = mask_file.stem

        mask_type = image_filename[6:]
        if mask_type not in head_parts:
            continue

        image_id = image_filename[:5].lstrip("0")
        image_id = image_id if image_id else "0"
        if image_id not in mask_dict.keys():
            mask_dict[image_id] = []

        mask_dict[image_id].append(mask_file)
    # mask_dict sample:
    # {
    #     '1': ['/path/to/00001_hair.png', '/path/to/00001_eye.png'],
    #     '2': ['/path/to/00002_hair.png'],
    #     '3': ['/path/to/00003_skin.png']
    # }
    return mask_dict

# PARTS = {
#     'background': 0,
#     'hair': 1,
#     'face': 2,
#     'neck': 3,
#     # Add other parts as needed
# }

def create_multiclass_segmaps(mask_files: Dict[str, List[Path]],
                              save_dir: Path,
                              classes,
                              composite_classes: Dict[str, List[str]]):
    logging.basicConfig(level=logging.INFO)
    logging.info(
        f"Creating segmentation masks and saving as PNG files in {save_dir}..."
    )
    # for i,(image_id, mask_paths) in enumerate(mask_files.items()):

    for image_id, mask_paths in tqdm(mask_files.items()):
        final_segmap = np.zeros((512, 512, len(classes) + 1), dtype=np.uint8)  # +1 for background

        for mask_file in mask_paths:
            mask_image = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
            mask_image = np.where(mask_image > 0, 1, 0)
            matched = False

            for class_idx, class_name in enumerate(classes):
                if class_name in composite_classes:
                    for sub_part in composite_classes[class_name]:
                        if f'_{sub_part}' in str(mask_file):
                            matched = True
                            final_segmap[:, :, class_idx] = np.maximum(final_segmap[:, :, class_idx], mask_image)
                else:
                    # Handle non-composite classes
                    if f'_{class_name}' in str(mask_file):
                        matched = True
                        final_segmap[:, :, class_idx] = np.maximum(final_segmap[:, :, class_idx], mask_image)

            # if not matched:
            #     print(f"No match found for {mask_file}")

        # Background handling: mark background where all other classes are absent
        background_index = len(classes)  # index for the background
        final_segmap[:, :, background_index] = np.all(final_segmap[:, :, :background_index] == 0, axis=2)

        final_segmap = cv2.resize(final_segmap, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        output_file = save_dir / f"{image_id}.png"
        cv2.imwrite(str(output_file), final_segmap)
        tqdm.write(f"Saved {output_file}", end="\r")

# def create_multiclass_segmaps(mask_files: Dict[str, List[Path]],
#                               save_dir: Path,
#                               classes,
#                               composite_classes: Dict[str, List[str]]):
#     logging.basicConfig(level=logging.INFO)
#     logging.info(
#         f"Creating segmentation masks and saving as PNG files in {save_dir}..."
#     )
#
#     for i,(image_id, mask_paths) in enumerate(mask_files.items()):
#       if i==0:
#         final_segmap = np.zeros((512, 512, len(classes) + 1), dtype=np.uint8)  # +1 for background
#
#         for mask_file in mask_paths:
#             mask_image = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
#             matched = False
#
#             for class_idx, class_name in enumerate(classes):
#                 if class_name in composite_classes:
#                     for sub_part in composite_classes[class_name]:
#                         if f'_{sub_part}' in str(mask_file):
#                             matched = True
#                             final_segmap[:, :, class_idx] = np.maximum(final_segmap[:, :, class_idx], mask_image)
#
#             if not matched:
#                 print(f"No match found for {mask_file}")
#
#         # Background handling: mark background where all other classes are absent
#         background_index = len(classes)  # index for the background
#         final_segmap[:, :, background_index] = np.all(final_segmap[:, :, :background_index] == 0, axis=2)
#
#         final_segmap = cv2.resize(final_segmap, (1024, 1024), interpolation=cv2.INTER_NEAREST)
#         output_file = save_dir / f"{image_id}.png"
#         cv2.imwrite(str(output_file), final_segmap)
#         tqdm.write(f"Saved {output_file}", end="\r")

# Example usage of the function
# You would need to define 'mask_files', 'save_dir', 'classes', and 'composite_classes' according to your data and requirements.

# def create_multiclass_segmaps(mask_files: Dict[str, List[Path]],
#                               save_dir: Path,
#                               classes,
#                               composite_classes: Dict[str, List[str]]):
#     logging.basicConfig(level=logging.INFO)
#     logging.info(
#         f"Creating segmentation masks and saving as PNG files in {save_dir}..."
#     )
#
#     PARTS = {class_name: f'_{class_name}' for class_name in classes}
#     print("PARTS: ", PARTS)
#
#     relevant_parts = set(PARTS.values())
#     for composite in composite_classes.values():
#         relevant_parts.update(f'_{sub_part}' for sub_part in composite)
#
#     for image_id, mask_paths in tqdm(mask_files.items()):
#         final_segmap = np.zeros((512, 512, len(classes) + 1), dtype=np.uint8)  # +1 for background
#
#         for mask_file in mask_paths:
#             if any(part in str(mask_file) for part in relevant_parts):
#                 mask_image = cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE)
#
#                 for class_idx, part in enumerate(PARTS.values()):
#                     if part in composite_classes:
#                         if any(f'_{sub_part}' in str(mask_file) for sub_part in composite_classes[part]):
#                             final_segmap[:, :, class_idx] = np.maximum(final_segmap[:, :, class_idx], mask_image)
#                     else:
#                         if part in str(mask_file):
#                             final_segmap[:, :, class_idx] = np.maximum(final_segmap[:, :, class_idx], mask_image)
#
#         # Background handling: mark background where all other classes are absent
#         background_index = len(classes)  # index for the background
#         final_segmap[:, :, background_index] = np.all(final_segmap[:, :, :background_index] == 0, axis=2)
#
#         final_segmap = cv2.resize(final_segmap, (1024, 1024), interpolation=cv2.INTER_NEAREST)
#         output_file = save_dir / f"{image_id}.png"
#         cv2.imwrite(str(output_file), final_segmap)
#         tqdm.write(f"Saved {output_file}", end="\r")






def create_multiclass_segmaps_pixel_wise(mask_files: t.Dict[str, t.List[Path]],
                              save_dir: Path,
                              classes,
                              composite_classes: t.Dict[str, t.List[str]]

                              ) :
    logging.info(
        f"Creating segmentation masks and saving as PNG files in {save_dir}..."
    )

    # for image_id, mask_paths in mask_files.items():
    #     mask_images = [cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) for mask_file in mask_paths]

    # Ensure 'classes' is loaded from your config or is defined somewhere
    PARTS = {class_name: {'suffix': f'_{class_name}', 'value': idx + 1} for idx, class_name in enumerate(classes)}
    print("PARTS: ",PARTS)

    for i, (image_id, mask_paths) in enumerate(mask_files.items()):
      # if i==10:
      if 1 == 1:
        # final_segmap = np.zeros((1024, 1024), dtype=np.uint8)
        final_segmap = np.zeros((512, 512), dtype=np.uint8)
        # print(" len image_id:", len(image_id))
        # print(" len mask_paths:", len(mask_paths))
        for part, class_info in PARTS.items():
            # print("------ part :", part)
            # print("------ class_info :", class_info)

            # Check if the class is composite
            if part in composite_classes:
                # print("          part is a composite class")
                composite_masks = []
                for sub_part in composite_classes[part]:
                    composite_masks.extend([mask for mask in mask_paths if f'_{sub_part}' in str(mask)])
                # print("ingredients for composite_mask:")
                # [print("   "+ str(p)) for p in composite_masks]

                part_masks = composite_masks
            else:
                # print("          part is NOT a composite class")
                # print("class_info['suffix']", class_info['suffix'])
                # print("mask_paths:")
                # [print("   " + str(p)) for p in mask_paths]
                part_masks = [mask for mask in mask_paths if class_info['suffix'] in str(mask)]
                # print("part_masks", part_masks)

            if not part_masks:
                continue

            mask_images = [  cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) for mask_file in part_masks ]
            # for img in mask_images:
            #     print(img.shape , np.unique(img))

            mask_images = np.array(mask_images)
            # print(mask_images.shape)
            # print(mask_images.shape)
            aggregate_lbls = mask_images.sum(axis=0)
            # print(aggregate_lbls.shape, np.unique(aggregate_lbls))
            aggregate_lbls[aggregate_lbls > 0] = class_info['value']
            # print(aggregate_lbls.shape, np.unique(aggregate_lbls))

            # Merge into the final segmentation map
            final_segmap += aggregate_lbls
            # take only neck intersection part. dismiss other rest of the neck.
            final_segmap[final_segmap == 2] = 0
            final_segmap[final_segmap == 3] = 2
            # print("final_segmap:", final_segmap.shape, np.unique(final_segmap))
            # cv2.imwrite('/home/enes/lab/head-segmentation/final_segmap.jpg', final_segmap)
            # temp=final_segmap.copy()
            # temp[temp == 0] = 60
            # temp[temp == 1] = 100
            # temp[temp == 2] = 150
            # temp[temp == 3] = 250
            # cv2.imwrite('/home/enes/lab/head-segmentation/final_segmap2.jpg', temp)

        # Resize final segmentation map using nearest neighbor interpolation
        final_segmap = cv2.resize(final_segmap, (1024, 1024), interpolation=cv2.INTER_NEAREST)

        output_file = save_dir / f"{image_id}.png"
        cv2.imwrite(str(output_file), final_segmap)
        tqdm.write(f"Saved {output_file}", end="\r")


def create_segmaps(mask_files: t.Dict[str, t.List[Path]], save_dir: Path) -> None:
    logging.info(
        f"Creating segmentation masks and saving as PNG files in {save_dir}..."
    )
    for image_id, mask_paths in mask_files.items():
        mask_images = [
            cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) for mask_file in mask_paths
        ]

        mask_images = np.array(mask_images)
        aggregate_lbls = mask_images.sum(axis=0)
        aggregate_lbls[aggregate_lbls > 0] = 1

        # Resize to image size
        segmap = cv2.resize(aggregate_lbls.astype(np.uint8), (1024, 1024))

        output_file = save_dir / f"{image_id}.png"
        cv2.imwrite(str(output_file), segmap)
        tqdm.write(f"Saved {output_file}", end="\r")


# python ./scripts/dataset/preprocess_raw_dataset.py --raw_dset_dir ./CelebAMask-HQ --output_dset_dir ./preprocessed_dset_dir

# python ./scripts/dataset/preprocess_raw_dataset.py --raw_dset_dir /home/enes/lab/CelebAMask-HQ --output_dset_dir /home/enes/lab/preprocessed_dset_dir



@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs"), config_name="training_experiment"
)
def main(configs) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    logging.info("Creating preprocessed dataset STARTED")
    # args = parse_args()

    preprocessed_dset_dir= configs["dataset_module"]["preprocessed_dset_dir"]
    raw_dset_dir= configs["dataset_module"]["raw_dset_dir"]

    preprocessed_dset_dir = Path(preprocessed_dset_dir)
    raw_dset_dir = Path(raw_dset_dir)


    new_images_dir, new_segmap_dir = create_structure(root_path=preprocessed_dset_dir)

    create_metadata_csv(
        attribute_txt=raw_dset_dir / "CelebAMask-HQ-attribute-anno.txt",
        output_dset_path=preprocessed_dset_dir,
    )

    copy_jpg_images(
        src_path=raw_dset_dir / "CelebA-HQ-img", dst_path=new_images_dir
    )
    mask_files = load_mask_files(src_path=raw_dset_dir / "CelebAMask-HQ-mask-anno")
    # create_segmaps(mask_files=mask_files, save_dir=new_segmap_dir)

    composite_classes=  configs["dataset_module"]["composite_classes"]
    print(type[composite_classes])
    print("composite_classes",composite_classes )
    classes= configs["dataset_module"]["classes"]
    create_multiclass_segmaps(mask_files=mask_files, save_dir=new_segmap_dir , classes=classes, composite_classes=composite_classes )

    logging.info("Creating preprocessed dataset FINISHED")


if __name__ == "__main__":
    main()


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
from utils import check_segmaps


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




def get_only_existing_masks(all_relevant_masks_for_this_class,all_masks_for_that_image ):
    composite_masks = []
    # print(all_masks_for_that_image)
    for mask in all_masks_for_that_image:
        first_underscore_index = str(mask).find('_')
        dot_index = str(mask).find('.')
        v=str(mask)[first_underscore_index + 1:dot_index]
        # v=str(mask)[last_underscore_index + 1:]
        if v in all_relevant_masks_for_this_class:
            composite_masks.append(mask)
    return composite_masks


def gather_relevant_masks(cls,composite_classes ):
    # composite_masks.extend([mask for mask in all_masks_for_that_image if f'_{sub_part}' in str(mask)])

    relevant_masks=[]
    if composite_classes is not None:
        if cls in composite_classes:
            for sub_class in composite_classes[cls]:
                relevant_masks.append(sub_class)
        else:
            relevant_masks.append(cls)
    if composite_classes is None:
        relevant_masks.append(cls)

    return relevant_masks

 #
 #
 # for main_cls, class_info in CLASSES.items():
 #            all_sub_cls_masks=[]
 #            # check if main cls is composite
 #            # if it is composite, collect all sub_masks in a list
 #            # if it is not composite collect that mask in a list

def create_relevant_classes_list( classes, composite_classes):
        relevant_classes=[]
        # (classes: head neck)
        if composite_classes is not None:
            for cls in classes:
                if cls in composite_classes:
                    for sub_class in composite_classes[cls]:
                        relevant_classes.append(sub_class)
                else:
                    relevant_classes.append(cls)
        if composite_classes is None:
            pass

        return relevant_classes



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


def load_mask_files(src_path: Path, relevant_mask_types) -> t.Dict[str, t.List[Path]]:
    logging.info("Loading mask files...")

    mask_dict = {}
    mask_files = sorted(list(src_path.rglob("*.png")))
    for mask_file in mask_files:
        image_filename = mask_file.stem

        mask_type = image_filename[6:]
        if mask_type not in  relevant_mask_types:
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

    #todo check if created masks may need to be in form of boolean and not uint8
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
        print("final_segmap.shape", final_segmap.shape)
        final_segmap = cv2.resize(final_segmap, (1024, 1024), interpolation=cv2.INTER_NEAREST)
        output_file = save_dir / f"{image_id}.png"
        cv2.imwrite(str(output_file), final_segmap)
        tqdm.write(f"Saved {output_file}", end="\r")


def create_multiclass_segmaps_pixel_wise(mask_files: t.Dict[str, t.List[Path]],
                              save_dir: Path,
                              classes,
                              composite_classes: t.Dict[str, t.List[str]]  ) :
    logging.info( f"Creating seg masks and saving as PNG files in {save_dir}..." )


    class_dict = {class_name: { 'value': idx + 1} for idx, class_name in enumerate(classes)}
    print(" ")
    print(" ")
    print("CLASSES: ", class_dict)

    for i, (image_id, all_masks_for_that_image) in enumerate(mask_files.items()):
      # if i==0:
        final_segmap = np.zeros((512, 512), dtype=np.uint8)
        # print("class_dict:", class_dict)

        for ii, (main_cls, class_info) in enumerate(class_dict.items()):
            # print("------ii: ",ii)
            # print("main_cls:", main_cls, "class_info:", class_info)
            all_relevant_masks_for_this_class=gather_relevant_masks(main_cls, composite_classes)
            # print("all_relevant_masks_for_this_class", all_relevant_masks_for_this_class)
            all_relevant_masks_for_this_class_and_exist_for_this_image=get_only_existing_masks(all_relevant_masks_for_this_class,all_masks_for_that_image )
            # print("all_relevant_masks_for_this_class_and_exist_for_this_image", all_relevant_masks_for_this_class_and_exist_for_this_image)
            # if 1==1:
            if all_relevant_masks_for_this_class_and_exist_for_this_image:
                mask_images = [  cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) for mask_file in all_relevant_masks_for_this_class_and_exist_for_this_image ]
                len_masks= len(mask_images)
                # print("len_masks:", len_masks)
                shape1= mask_images[0].shape
                # [print(e.shape) for e in mask_images]
                mask_images = np.array(mask_images)
                shape2 = mask_images.shape
                # print("merged mask_images shape:", mask_images.shape)
                # print("mask_images shape", mask_images.shape)
                aggregate_lbls = mask_images.sum(axis=0)
                # print("aggregate shape", aggregate_lbls.shape)
                # print("unique aggregate_lbls", np.unique(aggregate_lbls))
                # print("aggregate_lbls shape:", aggregate_lbls.shape)
                try:
                     # print("final_segmap shape:", final_segmap.shape)
                     aggregate_lbls[aggregate_lbls > 0] = class_info['value']
                     final_segmap += aggregate_lbls
                     # print("afer += final_segmap shape:", final_segmap.shape)
                # except ValueError:
                except Exception as e:

                    print(e)
                    # print("ValueError catched")
                    #
                    #
                    # print("len_masks:", len_masks)
                    # print("i:", i, "image_id:", image_id)
                    # print("main_cls:", main_cls, "class_info:",class_info )
                    # print("all_relevant_masks_for_this_class", all_relevant_masks_for_this_class)
                    # print("all_relevant_masks_for_this_class_and_exist_for_this_image", all_relevant_masks_for_this_class_and_exist_for_this_image)
                    # print("shape1:", shape1)
                    # print("shape2:", shape2)
                    # print("aggregate_lbls.shape:",aggregate_lbls.shape)

                # take only neck intersection part. dismiss other rest of the neck.
                # final_segmap[final_segmap == 2] = 0
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
    composite_classes = configs["dataset_module"]["composite_classes"]
    classes = configs["dataset_module"]["classes"]
    preprocessed_dset_dir= configs["dataset_module"]["preprocessed_dset_dir"]
    raw_dset_dir= configs["dataset_module"]["raw_dset_dir"]

    print("    classes", classes)
    print("    composite_classes", composite_classes)

    relevant_mask_types = []
    print(type(composite_classes))
    for class_name in classes:
        if composite_classes is not None :

            if class_name in composite_classes:

                relevant_mask_types.extend( composite_classes[class_name])
            else:
                relevant_mask_types.extend([class_name])
        else:
             relevant_mask_types.extend([class_name])


    print("    relevant_mask_types", relevant_mask_types)


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

    mask_files = load_mask_files(src_path=raw_dset_dir / "CelebAMask-HQ-mask-anno", relevant_mask_types=relevant_mask_types)
    # create_segmaps(mask_files=mask_files, save_dir=new_segmap_dir)

    # create_multiclass_segmaps
    create_multiclass_segmaps_pixel_wise(mask_files=mask_files, save_dir=new_segmap_dir , classes=classes, composite_classes=composite_classes )
    # segmap_sample_path = "/home/enes/lab/preprocessed_dset_dir/segmaps/0.png"
    # check_segmaps(segmap_sample_path)
    logging.info("Creating preprocessed dataset FINISHED")


if __name__ == "__main__":
    main()

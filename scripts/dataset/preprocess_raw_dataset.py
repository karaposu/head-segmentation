import argparse
import logging
import shutil
import typing as t
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess dataset.")

    # fmt: off
    parser.add_argument("--raw_dset_dir", "-r", type=Path, required=True, help="Raw CelebA dataset directory.")
    parser.add_argument("--output_dset_dir", "-o", type=Path, help="Preprocessed CelebA dataset directory.")
    # fmt: on

    return parser.parse_args()


def create_structure(root_path: Path) -> t.Tuple[Path, Path]:
    logging.info(f"Creating dataset structure with root dir {root_path}...")
    images_path = root_path / "images"
    images_path.mkdir(parents=True, exist_ok=False)

    segmaps_path = root_path / "segmaps"
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
    attribs_df.to_csv(str(metadata_csv), index=False)


def copy_jpg_images(src_path: Path, dst_path: Path) -> None:
    logging.info(f"Copying JPG images from {src_path} to {dst_path}...")
    for image_file in src_path.glob("*.jpg"):
        shutil.copy(image_file, dst_path / image_file.name)


def load_mask_files(src_path: Path) -> t.Dict[str, t.List[Path]]:
    logging.info("Loading mask files...")
    head_parts = {
        "ear_r",
        "eye_g",
        "hair",
        "hat",
        "l_brow",
        "l_ear",
        "l_eye",
        "l_lip",
        "mouth",
        "nose",
        "r_brow",
        "r_ear",
        "r_eye",
        "skin",
        "u_lip",
    }

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

PARTS = {
    'background': 0,
    'hair': 1,
    'face': 2,
    'neck': 3,
    # Add other parts as needed
}

def create_multiclass_segmaps(mask_files: t.Dict[str, t.List[Path]], save_dir: Path) -> None:
    logging.info(
        f"Creating segmentation masks and saving as PNG files in {save_dir}..."
    )
    for image_id, mask_paths in mask_files.items():
        final_segmap = np.zeros((1024, 1024), dtype=np.uint8)  # Initial size, can be adjusted

        for part, class_value in PARTS.items():
            if part == 'background':
                continue  # Skip background here, it's handled by default initialization

            part_masks = [mask for mask in mask_paths if part in str(mask)]
            if not part_masks:
                continue

            mask_images = [
                cv2.imread(str(mask_file), cv2.IMREAD_GRAYSCALE) for mask_file in part_masks
            ]

            mask_images = np.array(mask_images)
            aggregate_lbls = mask_images.sum(axis=0)
            aggregate_lbls[aggregate_lbls > 0] = class_value  # Assign class value

            final_segmap[aggregate_lbls == class_value] = class_value  # Merge into final segmentation map

        # Resize final segmentation map (if needed)
        final_segmap = cv2.resize(final_segmap, (1024, 1024))

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

def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")

    logging.info("Creating preprocessed dataset STARTED")
    args = parse_args()

    new_images_dir, new_segmap_dir = create_structure(root_path=args.output_dset_dir)

    create_metadata_csv(
        attribute_txt=args.raw_dset_dir / "CelebAMask-HQ-attribute-anno.txt",
        output_dset_path=args.output_dset_dir,
    )

    copy_jpg_images(
        src_path=args.raw_dset_dir / "CelebA-HQ-img", dst_path=new_images_dir
    )
    mask_files = load_mask_files(src_path=args.raw_dset_dir / "CelebAMask-HQ-mask-anno")
    # create_segmaps(mask_files=mask_files, save_dir=new_segmap_dir)
    create_multiclass_segmaps(mask_files=mask_files, save_dir=new_segmap_dir)
    logging.info("Creating preprocessed dataset FINISHED")


if __name__ == "__main__":
    main()

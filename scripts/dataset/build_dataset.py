import argparse
import logging
import random
import shutil
import typing as t
from pathlib import Path
import os
import hydra

from pathlib import Path
import typing as t


def load_images(dset_path: Path,segmaps_path ) -> t.List[str]:
    images_path = dset_path / "images"
    # segmaps_path = dset_path / "masks"
    segmaps_path="/home/enes/lab/preprocessed_dset_dir/segmaps"
    segmaps_path=Path(segmaps_path)

    image_files = []

    for img_path in images_path.glob("*.jpg"):
        # Construct the path for the corresponding mask file
        mask_path = segmaps_path / img_path.with_suffix('.png').name

        # Check if the mask file exists
        if mask_path.exists():
            # Add the image file to the list if its corresponding mask exists
            image_files.append(str(img_path))

    return image_files


# def load_images(dset_path: Path) -> t.List[str]:
#     return list((dset_path / "images").glob("*.jpg"))


def split_dataset(
    dset_images: list,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
) -> t.Tuple[list, list, list]:
    random.shuffle(dset_images)



    dset_size = len(dset_images)

    train_last_index = int(dset_size * train_frac)

    print("dset_size:", dset_size)
    print("train_last_index:", train_last_index)
    
    train_dset = dset_images[:train_last_index]
    print("len train_dset:", len(train_dset))



    val_last_index = train_last_index + int(dset_size * val_frac)
    print("val_last_index:", val_last_index)

    val_dset = dset_images[train_last_index:val_last_index]
    print("len val_dset:", len(val_dset))


    test_dset = dset_images[val_last_index:]

    return train_dset, val_dset, test_dset


def create_dataset_structure(output_dir: Path) -> None:
    train_dir = output_dir / "train"
    val_dir = output_dir / "val"
    test_dir = output_dir / "test"

    for dataset_dir in [train_dir, val_dir, test_dir]:
        images_dir = dataset_dir / "images"
        segmaps_dir = dataset_dir / "segmaps"

        images_dir.mkdir(parents=True, exist_ok=False)
        segmaps_dir.mkdir(parents=True, exist_ok=False)


def copy_images_to_dataset_dir(
    train_dset: list,
    val_dset: list,
    test_dset: list,
    output_dset_root: Path,
) -> None:
    for image_file in train_dset:
        copy_sample(image_file, output_dset_root / "train")

    for image_file in val_dset:
        copy_sample(image_file, output_dset_root / "val")

    for image_file in test_dset:
        copy_sample(image_file, output_dset_root / "test")


def copy_sample(image_file: Path, dset_path: Path) -> None:
    if isinstance(image_file, str):
        image_file = Path(image_file)
    shutil.copy(image_file, dset_path / "images" / image_file.name)

    segmap_path = image_file.parent.parent / "segmaps" / f"{image_file.stem}.png"
    shutil.copy(segmap_path, dset_path / "segmaps" / segmap_path.name)

@hydra.main(
    config_path=os.path.join(os.getcwd(), "configs"), config_name="training_experiment"
)
def main(configs) -> None:
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s"
    )

    logging.info("Building dataset STARTED")
    # args = parse_args()
    # "--preprocessed_dset", "-d", type = Path, required = True, help = "Preprocessed CelebA dataset directory.")
    # parser.add_argument("--output_dset_root"
    #
    output_dset_dir = configs["dataset_module"]["output_dset_dir"]
    preprocessed_dset_dir = configs["dataset_module"]["preprocessed_dset_dir"]

    preprocessed_dset_dir = Path(preprocessed_dset_dir)
    output_dset_dir = Path(output_dset_dir)
    
    image_files = load_images(preprocessed_dset_dir, "")
    print("len(image_files): ", len(image_files))
    train_dset, val_dset, test_dset = split_dataset(image_files)

    create_dataset_structure(output_dset_dir)

    copy_images_to_dataset_dir(
        train_dset=train_dset,
        val_dset=val_dset,
        test_dset=test_dset,
        output_dset_root=output_dset_dir,
    )
    shutil.copy(
        src=preprocessed_dset_dir/ "metadata.csv",
        dst=output_dset_dir / "metadata.csv",
    )

    logging.info("Building dataset FINISHED")


if __name__ == "__main__":
    main()

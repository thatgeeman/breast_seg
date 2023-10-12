import os
from dotenv import load_dotenv
from fastcore.xtras import Path
from numpy.typing import ArrayLike
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from argparse import ArgumentParser

load_dotenv()


def write_nifti_from_npy(pth: Path, threshold: float, save_path: Path):
    """Read nifti file"""
    if isinstance(pth, str):
        pth = Path(pth)
    if isinstance(save_path, str):
        save_path = Path(save_path)
    assert pth.exists(), f"Path: {pth} does not exist"
    image_arr = (np.load(pth) > threshold).astype(np.float32)
    image_sitk = sitk.GetImageFromArray(image_arr)
    sitk.WriteImage(image_sitk, save_path.as_posix(), imageIO="NiftiImageIO")


if __name__ == "__main__":
    # python predict.py --target-tissue breast --image $INPUT_ISPY1_NPY --save-masks-dir $DIRECTORY_TO_SAVE_PREDICTIONS --model-save-path $TRAINED_MODEL_PATH
    args = ArgumentParser()
    args.add_argument(
        "-threshold",
        default=0.5,
        required=False,
        type=float,
        help="Threshold for the probability masks",
    )
    args = args.parse_args()
    threshold = args.threshold
    print(f"Using threshold of {threshold} to create binary masks")
    ispy_npy_path = Path(os.environ["DIRECTORY_TO_SAVE_PREDICTIONS"])
    save_npy_path = Path(os.environ["INPUT_ISPY1_NPY"])
    file_end_slug = "DCE_0001_N3_zscored.npy"
    for ispy_npy_path_file in tqdm(ispy_npy_path.rglob("*.npy")):
        save_npy_path_file = (
            ispy_npy_path
            / f"binarymask_{ispy_npy_path_file.name.replace('npy', 'nii.gz')}"
        )
        # print(save_npy_path_file)
        write_nifti_from_npy(ispy_npy_path_file, threshold, save_npy_path_file)

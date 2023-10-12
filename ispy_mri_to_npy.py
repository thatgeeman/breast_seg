import os
from dotenv import load_dotenv
from fastcore.xtras import Path
from numpy.typing import ArrayLike
import numpy as np
import SimpleITK as sitk
from tqdm import tqdm

load_dotenv()


def read_nifti(
    pth: Path,
    as_array: bool = True,
) -> ArrayLike:
    """Read nifti file"""
    if isinstance(pth, str):
        pth = Path(pth)
    assert pth.exists(), f"Path: {pth} does not exist"
    image = sitk.ReadImage(pth.as_posix(), sitk.sitkFloat32, imageIO="NiftiImageIO")
    if as_array:
        arr = sitk.GetArrayFromImage(image)
        return arr
    return image


if __name__ == "__main__":
    # python predict.py --target-tissue breast --image $INPUT_ISPY1_NPY --save-masks-dir $DIRECTORY_TO_SAVE_PREDICTIONS --model-save-path $TRAINED_MODEL_PATH
    ispy_path = Path(os.environ["INPUT_ISPY1"])
    save_npy_path = Path(os.environ["INPUT_ISPY1_NPY"])
    file_end_slug = "DCE_0001_N3_zscored.nii.gz"
    for ispy_dir in tqdm(ispy_path.ls()):
        ispy_file_path = ispy_dir / f"{ispy_dir.name}_{file_end_slug}"
        image_arr = read_nifti(ispy_file_path)
        file_end_slug_npy = file_end_slug.replace(".nii.gz", ".npy")
        save_npy_path_file = save_npy_path / f"{ispy_dir.name}_{file_end_slug_npy}"
        np.save(save_npy_path_file, image_arr)

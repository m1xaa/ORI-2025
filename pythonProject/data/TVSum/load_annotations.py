import os.path

import h5py

import constants


def load_annotations(root: str):
    path = os.path.join(root, constants.TVSUM_ANNO_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"TVSum .mat not found at: {path}")

    with h5py.File(path, "r") as f:
        print(f)


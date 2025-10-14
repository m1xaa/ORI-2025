import math
import os.path
import h5py
import numpy as np

import constants


def _hdf5_char_array_to_str(char_array):
    a = np.array(char_array).astype(np.uint16).flatten()
    return ''.join(chr(int(c)) for c in a)


def load_annotations(root: str):
    path = os.path.join(root, constants.TVSUM_ANNO_FILENAME)
    if not os.path.exists(path):
        raise FileNotFoundError(f"TVSum .mat not found at: {path}")

    annotations = {}

    with h5py.File(path, "r") as file:
        tv_sum = file['tvsum50']
        videos = tv_sum['video']
        num_videos = videos.shape[0]

        for i in range(num_videos):
            metadata = {k: file[tv_sum[k][i, 0]] for k in tv_sum.keys()}
            video_id = _hdf5_char_array_to_str(metadata["video"])
            title = _hdf5_char_array_to_str(metadata["title"])
            category = _hdf5_char_array_to_str(metadata["category"])
            length = float(np.array(metadata["length"]))
            n_frames = float(np.array(metadata["nframes"]))
            fps = math.ceil(n_frames / length) if length > 0 else 25
            user_anno = np.array(metadata["user_anno"])
            gt_score = np.array(metadata["gt_score"]).reshape(-1)

            annotations[video_id] = {
                "user_scores": user_anno,
                "gt_score": gt_score,
                "fps": fps,
                "n_frames": n_frames,
                "length": length,
                "title": title,
                "category": category,
            }

    return annotations

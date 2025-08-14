"""PoseDatasetAngleWin
----------------------
Each sample is a *window* (default ``50`` frames) extracted from a 16‑angle
``.npz`` file.  A companion pickle stores the true clip length which we embed
as ``total_frames`` so that MMAction2's transforms behave correctly.

Files now contain ``cos`` and ``sin`` arrays in ``[-1, 1]`` rather than raw
angles.  These are stacked as the channel dimension yielding tensors shaped
``(C, T, V, M) = (2, window, 16, 1)``.
"""

import numpy as np
import mmengine.fileio as fileio
from mmengine.logging import print_log
from mmaction.registry import DATASETS
from mmaction.datasets.base import BaseActionDataset


@DATASETS.register_module()
class PoseDatasetAngleWin(BaseActionDataset):
    """16-angle windows prepared for ST-GCN (C,T,V,M)."""

    def __init__(
        self,
        ann_file: str,
        pipeline,
        window: int = 50,
        lengths_pkl: str | None = None,
        split=None,                       # not used
        **kwargs,
    ):
        self.window = window
        self.length_map = fileio.load(lengths_pkl) if lengths_pkl else {}

        # Keep MMAction2 defaults, but disable short-clip filter (we pad)
        super().__init__(
            ann_file,
            pipeline,
            filter_cfg=None,
            lazy_init=False,
            **kwargs)

    # ------------------------------------------------------------------
    # Required API
    # ------------------------------------------------------------------
    def filter_data(self):
        """No filtering – keep every annotation in ann_file."""
        return self.data_list

    def load_data_list(self):
        data = fileio.load(str(self.ann_file))        # list[dict]
        assert isinstance(data, list) and data, (
            f'{self.ann_file} must contain a non-empty list')

        for d in data:
            # Store the original clip length for reference but set
            # ``total_frames`` to the window length. This avoids
            # mismatches in transforms such as ``UniformSampleFrames``
            # which expect the number of frames to match the tensor
            # shape of ``keypoint`` (always ``self.window`` for this
            # dataset).
            d['orig_total_frames'] = self.length_map.get(d['frame_dir'])
            d['total_frames'] = self.window
        print_log(f'Loaded {len(data):,} samples from {self.ann_file}', 'current')
        return data

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _load_keypoint(self, sample: dict) -> np.ndarray:
        """Return ``cos``/``sin`` as ``(C,T,V,M) = (2, window, 16, 1)``."""
        npz = np.load(sample['frame_dir'])
        cos = npz['cos']
        sin = npz['sin']
        arr = np.stack([cos, sin], axis=-1)             # (T, 16, 2)
        if np.isnan(arr).any():
            print("NaN in raw arr for", sample['frame_dir'])
        i0, W, T = sample['i0'], self.window, arr.shape[0]
        # -------- extract / pad one window --------
        if i0 == -1:                                  # short clip → centre-pad
            pre = (W - T) // 2
            post = W - T - pre
            win = np.pad(arr, ((pre, post), (0, 0), (0, 0)), mode='edge')
        else:                                         # normal window
            end = i0 + W
            if end > T:                               # overflow → pad tail
                pad = end - T
                win = np.pad(arr[i0:], ((0, pad), (0, 0), (0, 0)), mode='edge')
            else:
                win = arr[i0:end]                     # (W, 16, 2)

        # -------- (W,16,2) -> (C,T,V,M) --------
        x = win.astype(np.float32)                    # (W, 16, 2)
        x = x[None, :, :, :]                          # (1, W, 16, 2)
        if np.isnan(x).any():
            print("NaN after processing in", sample['frame_dir'])
        return x

    def prepare_data(self, idx: int):
        info = super().get_data_info(idx)
        keypt = self._load_keypoint(info)

        results = dict(
            keypoint=keypt,
            label=int(info['label']),
            total_frames=info.get('total_frames', self.window))

        return self.pipeline(results)

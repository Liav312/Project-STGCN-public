from __future__ import annotations

import copy
from mmcv.transforms import Compose
from mmaction.registry import DATASETS
from .pose_dataset_anglewin import PoseDatasetAngleWin

@DATASETS.register_module()
class PoseDatasetTwoView(PoseDatasetAngleWin):
    """Return two augmented views of each window."""

    def __init__(self, ann_file, pipeline, window=50, lengths_pkl=None, **kwargs):
        # Build our own pipeline for each view
        super().__init__(ann_file=ann_file, pipeline=[], window=window,
                         lengths_pkl=lengths_pkl, **kwargs)
        self.view_pipeline = Compose(pipeline)

    def prepare_data(self, idx: int):
        info = super().get_data_info(idx)
        keypt = self._load_keypoint(info)
        base = dict(keypoint=keypt, label=int(info['label']),
                    total_frames=self.window)
        v1 = self.view_pipeline(copy.deepcopy(base))
        v2 = self.view_pipeline(copy.deepcopy(base))
        # ``PackActionInputs`` in ``view_pipeline`` returns a single
        # ``ActionDataSample`` for each view. Downstream components expect
        # one ``ActionDataSample`` per sample, so keep only the first view's
        # data sample here.
        return dict(
            inputs=[v1['inputs'], v2['inputs']],
            data_samples=v1['data_samples'])

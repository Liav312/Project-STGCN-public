# Copyright (c) OpenMMLab. All rights reserved.
from __future__ import annotations
from typing import Tuple

import torch
from mmaction.registry import MODELS
from mmaction.models.recognizers import RecognizerGCN

import torch.nn.functional as F
from mmaction.structures import ActionDataSample
@MODELS.register_module(name='ContrastiveRecognizerGCN')
class ContrastiveRecognizerGCN(RecognizerGCN):
    """RecognizerGCN that accepts TWO augmented clips and returns TWO features.

    Expected `inputs` tensor shape from PoseDatasetTwoView:
        (2, B, num_clips, num_person, clip_len, num_joints, C)
        └─ view-0   view-1
    """

    def extract_feat(self,
                     inputs: torch.Tensor,
                     stage: str = 'backbone',
                     **kwargs) -> Tuple:

        # --------------------------------------------------
        # 1. split the two views
        # --------------------------------------------------
        if isinstance(inputs, torch.Tensor):
            # DataPreprocessor stacks the batch as (B, 2, ...).  Support both
            # this format and the (2, B, ...) format documented above.
            if inputs.size(0) == 2:
                view0, view1 = inputs[0], inputs[1]
            else:
                assert inputs.size(1) == 2, (
                    'Expect 2 views in dimension 1 or 0')
                view0, view1 = inputs[:, 0], inputs[:, 1]
        else:
            view0, view1 = inputs      # each is (B, num_clips, Np, T, V, C)

        # --------------------------------------------------
        # 2. use parent implementation to get one feature per view
        #    we CANNOT call super().extract_feat() because it expects
        #    (B, num_clips, …) not (2, B, …).  So we in-line the code.
        # --------------------------------------------------
        def _gcn_forward(x):
            bs, nc = x.shape[:2]               # B, num_clips
            x = x.reshape((bs * nc, ) + x.shape[2:])
            if x.dim() == 6 and x.size(2) == 1:    # dummy clip dim
                x = x.squeeze(2)
            return self.backbone(x)               # (B*nc, 1, C, T, V)

        feat0 = _gcn_forward(view0)               # (B*nc, 1, C, T, V)
        feat1 = _gcn_forward(view1)

        # reunite them as tuple for the head
        loss_predict_kwargs = dict()              # nothing extra
        return (feat0, feat1), loss_predict_kwargs


        # add inside the class ----------------------------------------


    def predict(self, inputs, data_samples=None, **kwargs):
        """Generate embeddings for each sample during validation/testing."""

        if isinstance(inputs, torch.Tensor):
            if inputs.size(0) == 2:
                view0, view1 = inputs[0], inputs[1]
            else:
                assert inputs.size(1) == 2, 'Expect 2 views in dimension 1 or 0'
                view0, view1 = inputs[:, 0], inputs[:, 1]
        else:
            view0, view1 = inputs

        def _embed(x):
            bs, nc = x.shape[:2]
            x = x.reshape((bs * nc, ) + x.shape[2:])
            if x.dim() == 6 and x.size(2) == 1:
                x = x.squeeze(2)
            feat = self.backbone(x)
            emb = self.cls_head(feat)        # (bs*nc, 128)
            emb = emb.view(bs, nc, -1).mean(dim=1)
            return emb

        emb0 = _embed(view0)
        emb1 = _embed(view1)
        emb = F.normalize((emb0 + emb1) / 2, dim=1)

        if data_samples is None:
            data_samples = [ActionDataSample() for _ in range(len(emb))]

        for ds, e in zip(data_samples, emb):
            ds.pred_emb = e.detach().cpu()

        return data_samples
    
    

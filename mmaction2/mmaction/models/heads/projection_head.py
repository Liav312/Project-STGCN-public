from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmaction.registry import MODELS
from .base import BaseHead
from metric_learning_losses import multi_similarity_loss
def ntxent(z1, z2, T=0.07):
    
    B = z1.size(0)
    z1 = F.normalize(z1, dim=-1)
    z2 = F.normalize(z2, dim=-1)
    logits = torch.matmul(z1, z2.T) / T
    labels = torch.arange(B, device=z1.device)
    return F.cross_entropy(logits, labels)

@MODELS.register_module()
class ProjectionHead(BaseHead):
    

    def __init__(
        self,
        in_channels: int = 256,
        out_channels: int = 128,
        normalize: bool = True,
        temperature: float = 0.07,
        loss_type: str = 'NTXent',
        init_cfg=None,
    ) -> None:
        super().__init__(num_classes=0,
                         in_channels=in_channels,
                         loss_cls=None,
                         average_clips=None,
                         init_cfg=init_cfg)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc   = nn.Linear(in_channels, out_channels, bias=False)
        self.norm = nn.LayerNorm(out_channels)
        self.normalize = normalize
        self.T = temperature
        self.loss_type = loss_type

    def _project(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc(x)
        x = self.norm(x)
        if self.normalize:
            x = F.normalize(x.float(), dim=1).to(x.dtype)
        return x

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        if isinstance(x, (tuple, list)):
            x = x[0]
        N, M, C, T, V = x.shape
        x = x.view(N * M, C, T, V)
        x = self.pool(x).view(N, M, C)
        x = x.mean(dim=1)
        return self._project(x)

    def loss(self,
             feats,
             data_samples=None,
             **kwargs):
        if self.loss_type=='NTXent':
            if isinstance(feats, tuple):
                z1, z2 = feats
            else:
                B = feats.size(0) // 2
            z1, z2 = feats[:B], feats[B:]
            z1 = self.forward(z1)
            z2 = self.forward(z2)
            loss = ntxent(z1, z2, T=self.T)
            return dict(loss_ntxent=loss)
        elif self.loss_type == 'MS':
                    x = self.forward(feats[0] if isinstance(feats, tuple) else feats)  # Single view for MS
                    labels = torch.cat([s.gt_label for s in data_samples])
                    loss = multi_similarity_loss(
                        x, labels, alpha=self.alpha, beta=self.beta, lambd=self.lambd, epsilon=self.epsilon,
                        distance=self.distance)
                    return dict(loss_ms=loss)

    def predict_by_feat(self, cls_scores, data_samples, **kwargs):
        # cls_scores here are actually embeddings (since no num_classes)
        for ds, emb in zip(data_samples, cls_scores):
            ds.pred_emb = emb  # Set as tensor (detach if needed: emb.detach())
        return data_samples
@MODELS.register_module()
class MetricProjectionHead(ProjectionHead):  # Or keep as ProjectionHead if not separating

    def __init__(self, loss_type='MS', distance='cosine', alpha=2.0, beta=50.0, lambd=0.5, epsilon=0.05, **kwargs):
            super().__init__(**kwargs)
            self.loss_type = loss_type
            self.distance = distance
            self.alpha = alpha
            self.beta = beta
            self.lambd = lambd
            self.epsilon = epsilon
    def set_epoch(self, epoch):
        self.current_epoch = epoch  # Then pass to loss
    def loss(self, feats, data_samples, epoch=None, **kwargs):
        x = self.forward(feats)  # For single-view
        labels = torch.cat([s.gt_label for s in data_samples])
        if self.loss_type == 'MS':
            loss = multi_similarity_loss(
                x, labels, alpha=self.alpha, beta=self.beta, lambd=self.lambd, epsilon=self.epsilon,
                distance=self.distance)
            return dict(loss_ms=loss)


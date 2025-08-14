from typing import Any

from mmengine.hooks import Hook
from mmengine.registry import HOOKS
import torch.nn as nn
from torch.nn import init

@HOOKS.register_module()
class FreezeFirstBlockHook(Hook):
    """Freeze data_bn and the first ST-GCN block for a few epochs."""

    def __init__(self, unfreeze_epoch: int = 10) -> None:
        self.unfreeze_epoch = unfreeze_epoch
        self._frozen = False

    def before_train(self, runner) -> None:
        self.reset_layers(runner.model)
        self.freeze_layers(runner.model)
        self._frozen = True

    def before_train_epoch(self, runner) -> None:
        if self._frozen and runner.epoch >= self.unfreeze_epoch:
            self.unfreeze_layers(runner.model)
            self._frozen = False

    def freeze_layers(self, model: Any) -> None:
        for name, param in model.backbone.named_parameters():
            if name.startswith('data_bn') or name.startswith('gcn.0'):
                param.requires_grad = False

    def unfreeze_layers(self, model: Any) -> None:
        for name, param in model.backbone.named_parameters():
            if name.startswith('data_bn') or name.startswith('gcn.0'):
                param.requires_grad = True

    def reset_layers(self, model: Any) -> None:
        """Re-init the first BN and first 1x1 conv for angle input."""
        bn = model.backbone.data_bn
        if isinstance(bn, nn.BatchNorm1d):
            init.ones_(bn.weight)
            init.zeros_(bn.bias)
        conv = model.backbone.gcn[0].gcn.conv
        init.kaiming_normal_(conv.weight, mode='fan_out')
        if conv.bias is not None:
            init.zeros_(conv.bias)

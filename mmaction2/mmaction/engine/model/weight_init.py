# Copyright (c) OpenMMLab. All rights reserved.
import math

import torch.nn as nn
from mmengine.model import BaseInit, update_init_info

import re
from typing import List, Optional

from mmaction.registry import WEIGHT_INITIALIZERS


def conv_branch_init(conv: nn.Module, branches: int) -> None:
    """Perform initialization for a conv branch.

    Args:
        conv (nn.Module): The conv module of a branch.
        branches (int): The number of branches.
    """

    weight = conv.weight
    n = weight.size(0)
    k1 = weight.size(1)
    k2 = weight.size(2)
    nn.init.normal_(weight, 0, math.sqrt(2. / (n * k1 * k2 * branches)))
    nn.init.constant_(conv.bias, 0)


@WEIGHT_INITIALIZERS.register_module('ConvBranch')
class ConvBranchInit(BaseInit):
    """Initialize the module parameters of different branches.

    Args:
        name (str): The name of the target module.
    """

    def __init__(self, name: str, **kwargs) -> None:
        super(ConvBranchInit, self).__init__(**kwargs)
        self.name = name

    def __call__(self, module) -> None:
        assert hasattr(module, self.name)

        # Take a short cut to get the target module
        module = getattr(module, self.name)
        num_subset = len(module)
        for conv in module:
            conv_branch_init(conv, num_subset)

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}'
        return info


@WEIGHT_INITIALIZERS.register_module('Pretrained', force=True)
class PretrainedInit:
    """Load a pretrained checkpoint with optional ignoring keys.

    Args:
        checkpoint (str): Path to the checkpoint to load.
        prefix (str, optional): Prefix for loading part of the checkpoint.
            Defaults to ''.
        map_location (str): Location to map checkpoint tensors. Defaults to
            'cpu'.
        strict (bool): Whether to strictly enforce that the keys in
            ``state_dict`` match the keys returned by the module's
            ``state_dict`` function. Defaults to False.
        ignore_keys (list[str], optional): Regex patterns of keys to ignore
            from the loaded state_dict. Defaults to None.
    """

    def __init__(self,
                 checkpoint: str,
                 prefix: str = '',
                 map_location: str = 'cpu',
                 strict: bool = False,
                 ignore_keys: Optional[List[str]] = None) -> None:
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.map_location = map_location
        self.strict = strict
        self.ignore_patterns = [re.compile(p) for p in (ignore_keys or [])]

    def __call__(self, module) -> None:
        from mmengine.runner.checkpoint import (
            _load_checkpoint_with_prefix, load_checkpoint, load_state_dict)

        if self.prefix:
            state_dict = _load_checkpoint_with_prefix(
                self.prefix, self.checkpoint, map_location=self.map_location)
        else:
            ckpt = load_checkpoint(
                module,
                self.checkpoint,
                map_location=self.map_location,
                strict=False,
                logger='current')
            state_dict = ckpt.get('state_dict', ckpt)

        if self.ignore_patterns:
            state_dict = {
                k: v
                for k, v in state_dict.items()
                if not any(p.search(k) for p in self.ignore_patterns)
            }

        load_state_dict(module, state_dict, strict=self.strict, logger='current')

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        return f'{self.__class__.__name__}: load from {self.checkpoint}'

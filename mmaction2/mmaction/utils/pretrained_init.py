from __future__ import annotations
import fnmatch
from mmengine.logging import print_log
from mmengine.runner.checkpoint import (_load_checkpoint_with_prefix,
                                        _load_checkpoint,
                                        load_state_dict)
from mmengine.model.weight_init import WEIGHT_INITIALIZERS, update_init_info

@WEIGHT_INITIALIZERS.register_module(name='Pretrained', force=True)
class PretrainedInit:
    """Extended initializer that supports ``strict`` and ``ignore_keys``.

    Args:
        checkpoint (str): Checkpoint file to load.
        prefix (str, optional): Prefix of sub-module in the checkpoint.
            Defaults to None.
        map_location (str): Device mapping for loading checkpoint. Defaults to
            ``'cpu'``.
        strict (bool): Whether to strictly enforce that the keys in ``state_dict``
            of checkpoint match the module's keys. Defaults to ``False``.
        ignore_keys (list[str], optional): Patterns of keys to ignore when
            loading checkpoint. Wildcards are supported. Defaults to ``None``.
    """

    def __init__(self,
                 checkpoint: str,
                 prefix: str | None = None,
                 map_location: str = 'cpu',
                 strict: bool = False,
                 ignore_keys: list[str] | None = None) -> None:
        self.checkpoint = checkpoint
        self.prefix = prefix
        self.map_location = map_location
        self.strict = strict
        self.ignore_keys = ignore_keys or []

    def __call__(self, module) -> None:
        if self.prefix:
            print_log(
                f'load {self.prefix} in model from: {self.checkpoint}',
                logger='current')
            state_dict = _load_checkpoint_with_prefix(
                self.prefix, self.checkpoint, map_location=self.map_location)
        else:
            print_log(f'load model from: {self.checkpoint}', logger='current')
            ckpt = _load_checkpoint(
                self.checkpoint, map_location=self.map_location)
            state_dict = ckpt['state_dict'] if 'state_dict' in ckpt else ckpt

        if self.ignore_keys:
            for pattern in self.ignore_keys:
                for k in list(state_dict.keys()):
                    if fnmatch.fnmatch(k, pattern):
                        state_dict.pop(k)

        load_state_dict(module, state_dict, strict=self.strict, logger='current')

        if hasattr(module, '_params_init_info'):
            update_init_info(module, init_info=self._get_init_info())

    def _get_init_info(self) -> str:
        info = f'{self.__class__.__name__}: load from {self.checkpoint}'
        return info

from mmengine.hooks import Hook
from mmaction.registry import HOOKS

@HOOKS.register_module()
class EpochToLossHook(Hook):
    def before_train_epoch(self, runner):
        model = runner.model.module if hasattr(runner.model, 'module') else runner.model
        if hasattr(model.cls_head, 'set_epoch'):
            model.cls_head.set_epoch(runner.epoch)  # Pass to head
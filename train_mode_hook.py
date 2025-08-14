from mmengine.hooks import Hook
from mmaction.registry import HOOKS

@HOOKS.register_module()
class TrainModeHook(Hook):
    def before_train_epoch(self, runner):
        runner.model.train()  # Ensure train mode
        runner.logger.info("Starting train epoch - model in train mode")
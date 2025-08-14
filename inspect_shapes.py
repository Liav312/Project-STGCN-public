

import sys
import os
import torch
from mmengine.config import Config
from mmaction.utils import register_all_modules
from mmaction.registry import MODELS, DATASETS


def main(cfg_path: str):
    cfg = Config.fromfile(cfg_path)
    register_all_modules()

    ckpt_path = cfg.get('load_from')
    if ckpt_path and os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        first_key = next(iter(state))
        print(f"First key in ckpt: {first_key} -> {state[first_key].shape}")
        cls_w = state.get('cls_head.fc.weight')
        if cls_w is not None:
            print(f"Classifier weight: {cls_w.shape}")
    else:
        print('Checkpoint not found')

    model = MODELS.build(cfg.model)
    print('Model data_bn running_mean size:', model.backbone.data_bn.running_mean.shape)

    dataset = DATASETS.build(cfg.train_dataloader.dataset)
    sample = dataset[0]
    inputs = torch.tensor(sample['inputs'], dtype=torch.float32)
    print('Raw sample shape:', inputs.shape)

    inputs = inputs.unsqueeze(0)
    inputs = inputs.permute(0, 4, 2, 3, 1)
    inputs = inputs.unsqueeze(1)
    print('Model input shape:', inputs.shape)

    try:
        out = model(inputs, stage='backbone')
        print('Backbone output shape:', out.shape)
    except Exception as e:
        print('Model forward failed:', e)


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python inspect_shapes.py <config>')
        sys.exit(1)
    main(sys.argv[1])

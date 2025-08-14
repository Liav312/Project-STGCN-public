ann_file_test = 'data/split/suemd-markless/suemd_test_windows.pkl'
ann_file_train = 'data/split/suemd-markless/suemd_train_windows.pkl'
ann_file_val = 'data/split/suemd-markless/suemd_val_windows.pkl'
common_dataset_cfg = dict(type='PoseDatasetTwoView', window=50)
custom_hooks = []
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmaction.models.recognizers.contrastive_recognizer',
        'graph_angles16_kinematic',
        'mmaction.datasets.pose_dataset_anglewin',
        'mmaction.datasets.pose_dataset_twoview',
        'mmaction.models.heads.projection_head',
        'angle_transforms',
        'prf1_metric',
        'utils.freeze_first_block_hook',
        'my_knn_metric',
    ])
dataset_type = 'PoseDatasetTwoView'
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'))
default_scope = 'mmaction'
fp16 = dict(loss_scale='dynamic')
init_cfg = dict(
    checkpoint=
    'work_dirs/ntu_angles16_onewin_finetune/best_acc_top1_epoch_18.pth',
    ignore_keys=[
        'cls_head.*',
    ],
    strict=True,
    type='Pretrained')
launcher = 'none'
load_from = 'work_dirs/suemd_ntxent_40epochs/epoch_40.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        base_channels=64,
        down_stages=[
            5,
        ],
        graph_cfg=dict(layout='angles16_kinematic', mode='spatial'),
        in_channels=1,
        inflate_stages=[
            5,
            8,
        ],
        num_stages=10,
        tcn_dropout=0.2,
        type='STGCN'),
    cls_head=dict(
        in_channels=256,
        normalize=True,
        out_channels=128,
        temperature=0.07,
        type='ProjectionHead'),
    type='ContrastiveRecognizerGCN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(lr=0.0003, type='AdamW', weight_decay=0.0001),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone.data_bn': dict(lr_mult=0.1),
            'backbone.gcn.0': dict(lr_mult=0.1)
        })),
    type='OptimWrapper')
param_scheduler = [
    dict(T_max=40, by_epoch=True, eta_min=1e-05, type='CosineAnnealingLR'),
]
randomness = dict(deterministic=True, diff_rank_seed=False, seed=42)
resume = False
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=256,
    dataset=dict(
        ann_file='data/split/suemd-markless/suemd_test_windows.pkl',
        lengths_pkl='data/split/suemd-markless/suemd_clip_lengths_test.pkl',
        pipeline=[
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='test',
        type='PoseDatasetTwoView',
        window=50),
    num_workers=20,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(k=5, metric='cosine', type='my_knn_metric.KNNMetric'),
]
train_cfg = dict(max_epochs=40, type='EpochBasedTrainLoop', val_interval=1)
train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        ann_file='data/split/suemd-markless/suemd_train_windows.pkl',
        lengths_pkl='data/split/suemd-markless/suemd_clip_lengths_train.pkl',
        pipeline=[
            dict(type='ContrastiveAug'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='train',
        type='PoseDatasetTwoView',
        window=50),
    num_workers=20,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=256,
    dataset=dict(
        ann_file='data/split/suemd-markless/suemd_val_windows.pkl',
        lengths_pkl='data/split/suemd-markless/suemd_clip_lengths_val.pkl',
        pipeline=[
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='val',
        type='PoseDatasetTwoView',
        window=50),
    num_workers=20,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(k=5, metric='cosine', type='my_knn_metric.KNNMetric'),
]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='TensorboardVisBackend'),
    ])
window_len = 50
work_dir = 'tmp_test_output'

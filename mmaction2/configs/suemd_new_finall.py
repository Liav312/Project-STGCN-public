ann_file_test = 'data/split/suemd-markless/suemd_test_windows.pkl'
ann_file_train = 'data/split/suemd-markless/suemd_train_windows.pkl'
ann_file_val = 'data/split/suemd-markless/suemd_val_windows.pkl'
common_dataset_cfg = dict(type='PoseDatasetAngleWin', window=50)
custom_hooks = [
    dict(type='EpochToLossHook'),
    dict(type='TrainModeHook'),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'mmaction.models.recognizers.contrastive_recognizer',
        'graph_angles16_kinematic3',
        'mmaction.datasets.pose_dataset_anglewin',
        'mmaction.datasets.pose_dataset_twoview',
        'mmaction.models.heads.projection_head',
        'angle_transforms',
        'prf1_metric',
        'utils.freeze_first_block_hook',
        'my_knn_metric',
        'metric_learning_losses',
        'epoch_hook',
        'class_balanced_sampler',
        'train_mode_hook',
    ])
dataset_type = 'PoseDatasetAngleWin'
default_hooks = dict(
    checkpoint=dict(interval=5, max_keep_ckpts=3, type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'))
default_scope = 'mmaction'
launcher = 'none'
load_from = 'work_dirs/suemd_finall_no_adaptive/epoch_40.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        base_channels=64,
        down_stages=[
            5,
        ],
        graph_cfg=dict(layout='angles16_kinematic', mode='stgcn_spatial'),
        in_channels=2,
        inflate_stages=[
            5,
            8,
        ],
        num_stages=10,
        tcn_dropout=0.2,
        type='STGCN'),
    cls_head=dict(
        alpha=2.0,
        beta=50.0,
        distance='cosine',
        epsilon=0.05,
        in_channels=256,
        lambd=0.5,
        loss_type='MS',
        normalize=True,
        out_channels=128,
        temperature=0.07,
        type='MetricProjectionHead'),
    init_cfg=dict(
        checkpoint='work_dirs/ntu_new_last_continue/epoch_55.pth',
        ignore_keys=[
            'cls_head.*',
        ],
        strict=False,
        type='Pretrained'),
    type='RecognizerGCN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(
        betas=(
            0.9,
            0.999,
        ), lr=0.001, type='AdamW', weight_decay=0.05),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone.data_bn': dict(lr_mult=0.05),
            'backbone.gcn.0': dict(lr_mult=0.05),
            'cls_head': dict(lr_mult=2.0)
        })),
    type='OptimWrapper')
param_scheduler = [
    dict(begin=0, by_epoch=True, end=3, start_factor=1e-05, type='LinearLR'),
    dict(
        T_max=37,
        begin=3,
        by_epoch=True,
        end=40,
        eta_min=1e-06,
        type='CosineAnnealingLR'),
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
        type='PoseDatasetAngleWin',
        window=50),
    num_workers=20,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(k=3, metric='cosine', type='KNNMetric'),
    dict(
        out_file_path='work_dirs/suemd_finall_no_adaptive/test_embeds.pkl',
        type='DumpResults'),
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
        type='PoseDatasetAngleWin',
        window=50),
    drop_last=False,
    num_workers=20)
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
        type='PoseDatasetAngleWin',
        window=50),
    num_workers=20,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(k=3, metric='cosine', type='KNNMetric'),
]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='TensorboardVisBackend'),
    ])
window_len = 50
work_dir = 'work_dirs/suemd_finall_no_adaptive'

ann_file_test = 'data/split/ntu/ntu_test_onewin.pkl'
ann_file_train = 'data/split/ntu/ntu_train_onewin.pkl'
ann_file_val = 'data/split/ntu/ntu_val_onewin.pkl'
common_dataset_cfg = dict(type='PoseDatasetAngleWin', window=50)
custom_hooks = [
    dict(type='FreezeFirstBlockHook', unfreeze_epoch=5),
]
custom_imports = dict(
    allow_failed_imports=False,
    imports=[
        'graph_angles16_kinematic3',
        'mmaction.datasets.pose_dataset_anglewin',
        'angle_transforms',
        'prf1_metric',
        'utils.freeze_first_block_hook',
    ])
dataset_type = 'PoseDatasetAngleWin'
default_hooks = dict(
    checkpoint=dict(
        interval=5,
        max_keep_ckpts=10,
        rule='greater',
        save_best='acc/top1',
        type='CheckpointHook'),
    logger=dict(interval=100, type='LoggerHook'))
default_scope = 'mmaction'
fp16 = dict(loss_scale='dynamic')
launcher = 'none'
load_from = 'work_dirs/ntu_new_last_continue/epoch_55.pth'
log_level = 'INFO'
model = dict(
    backbone=dict(
        graph_cfg=dict(layout='angles16_kinematic', mode='stgcn_spatial'),
        in_channels=2,
        tcn_dropout=0.2,
        type='STGCN'),
    cls_head=dict(
        in_channels=256,
        label_smooth_eps=0.1,
        loss_cls=dict(type='CrossEntropyLoss'),
        num_classes=120,
        type='GCNHead'),
    init_cfg=dict(
        checkpoint=
        'pretrain/stgcn_8xb16-joint-u100-80e_ntu120-xsub-keypoint-3d.pth',
        ignore_keys=[
            'backbone.data_bn',
            'backbone.gcn.0.gcn.conv',
            'backbone.gcn.*.gcn.PA',
            'backbone.gcn.*.gcn.A',
        ],
        prefix='',
        strict=False,
        type='Pretrained'),
    type='RecognizerGCN')
optim_wrapper = dict(
    clip_grad=dict(max_norm=1.0),
    optimizer=dict(
        lr=0.05, momentum=0.9, nesterov=True, type='SGD', weight_decay=0.0005),
    paramwise_cfg=dict(
        custom_keys=dict({
            'backbone': dict(lr_mult=0.2),
            'backbone.data_bn': dict(lr_mult=1.0),
            'backbone.gcn.0': dict(lr_mult=1.0),
            'cls_head.fc': dict(lr_mult=1.5)
        })),
    type='OptimWrapper')
param_scheduler = [
    dict(
        begin=0,
        by_epoch=True,
        end=30,
        gamma=0.1,
        milestones=[
            0,
            15,
        ],
        type='MultiStepLR'),
]
randomness = dict(deterministic=False, diff_rank_seed=False, seed=None)
resume = True
test_cfg = dict(type='TestLoop')
test_dataloader = dict(
    batch_size=256,
    dataset=dict(
        ann_file='data/split/ntu/ntu_test_onewin.pkl',
        lengths_pkl='data/split/ntu/ntu_clip_lengths_test.pkl',
        pipeline=[
            dict(
                clip_len=50,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='test',
        type='PoseDatasetAngleWin',
        window=50),
    num_workers=20,
    sampler=dict(shuffle=False, type='DefaultSampler'))
test_evaluator = [
    dict(
        metric_options=dict(top_k_accuracy=dict(topk=(1, ))),
        type='AccMetric'),
    dict(type='ConfusionMatrix'),
    dict(type='PRF1AUCMetric'),
    dict(
        out_file_path='work_dirs/test_results/results.pkl',
        type='DumpResults'),
]
train_cfg = dict(max_epochs=60, type='EpochBasedTrainLoop', val_interval=5)
train_dataloader = dict(
    batch_size=256,
    dataset=dict(
        ann_file='data/split/ntu/ntu_train_onewin.pkl',
        lengths_pkl='data/split/ntu/ntu_clip_lengths_train.pkl',
        pipeline=[
            dict(clip_len=50, type='UniformSampleFrames'),
            dict(prob=0.5, type='RandomFlipAngles'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='train',
        type='PoseDatasetAngleWin',
        window=50),
    num_workers=20,
    sampler=dict(shuffle=True, type='DefaultSampler'))
val_cfg = dict(type='ValLoop')
val_dataloader = dict(
    batch_size=256,
    dataset=dict(
        ann_file='data/split/ntu/ntu_val_onewin.pkl',
        lengths_pkl='data/split/ntu/ntu_clip_lengths_val.pkl',
        pipeline=[
            dict(
                clip_len=50,
                num_clips=1,
                test_mode=True,
                type='UniformSampleFrames'),
            dict(num_person=1, type='FormatGCNInput'),
            dict(type='PackActionInputs'),
        ],
        split='val',
        type='PoseDatasetAngleWin',
        window=50),
    num_workers=20,
    sampler=dict(shuffle=False, type='DefaultSampler'))
val_evaluator = [
    dict(
        metric_options=dict(top_k_accuracy=dict(topk=(1, ))),
        type='AccMetric'),
    dict(type='ConfusionMatrix'),
    dict(type='PRF1AUCMetric'),
]
visualizer = dict(
    type='ActionVisualizer',
    vis_backends=[
        dict(type='TensorboardVisBackend'),
    ])
window_len = 50
work_dir = 'work_dirs/ntu_new_last_continue'

# _base_ = [
#     '../_base_/models/retinanet_r50_fpn.py',
#     '../_base_/datasets/coco_detection.py',
#     '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py',
#     './retinanet_tta.py'
# ]

_base_ = [
    '../mmdetection/configs/retinanet/retinanet_r50_fpn_1x_coco.py'
]

### Setting ###
# dataset 설정을 해줍니다.
data_root='../../dataset/'
k='0'
epoch = 10
batch_size = 16
resize = (1024, 1024)

### Model ###
model = dict(
    type='RetinaNet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # mean=[123.675, 116.28, 103.53],   # origin
        # std=[58.395, 57.12, 57.375],
        mean=[124.338, 118.218, 110.6955], 
        std=[60.333, 58.956, 60.996],
        bgr_to_rgb=True,
        pad_mask=True,
        pad_size_divisor=32),
    bbox_head=dict(
        type='RetinaHead',
        num_classes=10,
    ),
)

### Dataset ###
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=resize, keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=resize,
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            # dict(type='Normalize', **img_norm_cfg),   version 3에서 사용하지 않습니다.
            # dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]

### Scheduler ###
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch, val_interval=5)
auto_scale_lr = dict(enable=True, base_batch_size=batch_size)

# /data/ephemeral/.vm/lib/python3.10/site-packages/mmdet/engine/hooks/
### Hook ###
custom_hooks = [
    dict(type='SubmissionHook', test_out_dir='submit'),
]

metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}
train_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'fold_{k}/stratified_train.json',
        data_prefix=dict(img='')))
val_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'fold_{k}/stratified_val.json',
        data_prefix=dict(img='')))

test_dataloader = dict(
    dataset=dict(
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'test.json',
        data_prefix=dict(img='')))

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + f'fold_{k}/stratified_val.json',
    metric='bbox',
    format_only=False,
    classwise=True,
    )
test_evaluator = dict(
    ann_file=data_root + 'test.json',
    classwise=True,
    )

### Checkpoint ###
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        save_best='auto'))

# 사용하는 모델의 pre-trainede된 checkpoint 경로/링크를 불러옵니다.
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/fp16/retinanet_r50_fpn_fp16_1x_coco/retinanet_r50_fpn_fp16_1x_coco_20200702-0dbfb212.pth'
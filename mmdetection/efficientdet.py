_base_ = [
    'projects/EfficientDet/configs/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco.py'
]
custom_imports = dict(
    imports=['projects.EfficientDet.efficientdet'], allow_failed_imports=False)

# _base_ = [
#     'mmdet::_base_/datasets/coco_detection.py',
#     'mmdet::_base_/schedules/schedule_1x.py',
#     'mmdet::_base_/default_runtime.py'
# ]

### Setting ###
image_size = 1024
batch_augments = [
    dict(type='BatchFixedSizePad', size=(image_size, image_size))
]
checkpoint = 'https://download.openmmlab.com/mmdetection/v3.0/efficientdet/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco_20230223_122457-e6f7a833.pth'  # noqa

data_root='../../dataset/'
k='1'
batch_size = 4


model = dict(
    type='EfficientDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # mean=[123.675, 116.28, 103.53],   # origin
        # std=[58.395, 57.12, 57.375],
        mean=[124.338, 118.218, 110.6955], 
        std=[60.333, 58.956, 60.996],
        bgr_to_rgb=True,
        pad_size_divisor=image_size,
        batch_augments=batch_augments),
    backbone=dict(
        type='EfficientNet',
        arch='b3',
        init_cfg=dict(
            type='Pretrained', prefix='backbone', checkpoint=checkpoint)),
    bbox_head=dict(
        type='EfficientDetSepBNHead',
        num_classes=10),
    )

train_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='RandomResize',
        scale=(image_size, image_size),
        ratio_range=(0.1, 2.0),
        keep_ratio=True),
    dict(type='RandomCrop', crop_size=(image_size, image_size)),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs')
]
test_pipeline = [
    dict(type='LoadImageFromFile', backend_args={{_base_.backend_args}}),
    dict(type='Resize', scale=(image_size, image_size), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
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
    num_workers=1,
    dataset=dict(
        data_root=data_root,
        ann_file=f'fold_{k}/stratified_train.json',
        metainfo=metainfo,
        pipeline=train_pipeline,
        data_prefix=dict(img='')))
val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        data_root=data_root,
        ann_file=f'fold_{k}/stratified_val.json',
        metainfo=metainfo,
        pipeline=test_pipeline,  
        data_prefix=dict(img='')))
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        data_root=data_root,
        ann_file=f'test.json',
        metainfo=metainfo,
        pipeline=test_pipeline,  
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

# learning policy
max_epochs = 300
param_scheduler = [
    dict(type='LinearLR', start_factor=0.1, by_epoch=False, begin=0, end=917),
    dict(
        type='CosineAnnealingLR',
        eta_min=0.0,
        begin=1,
        T_max=299,
        end=300,
        by_epoch=True,
        convert_to_iter_based=True)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

vis_backends = []
visualizer = dict(
    type='DetLocalVisualizer', vis_backends=vis_backends, name='visualizer')

### Checkpoint ###
default_hooks = dict(
    checkpoint=dict(
        by_epoch=False, 
        type="CheckpointHook",
        rule="greater",
        save_best="coco/bbox_mAP_50",
        interval=-1,
        max_keep_ckpts=3,
    )
)

### Hook ###
custom_hooks = [
    dict(type='SubmissionHook', test_out_dir='submit'),
]

# load_from = "https://download.openmmlab.com/mmdetection/v3.0/efficientdet/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco/efficientdet_effb3_bifpn_8xb16-crop896-300e_coco_20230223_122457-e6f7a833.pth"
# load_from = "efficientdet-d2.pth"
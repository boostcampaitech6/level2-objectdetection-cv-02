_base_ = ['projects/CO-DETR/configs/codino/co_dino_5scale_swin_l_16xb1_1x_coco.py']

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window12_384_22k.pth'  # noqa

### Setting ###
data_root='/data/ephemeral/home/dataset/'
k='1'
batch_size = 1
num_dec_layer = 6
loss_lambda = 2.0
num_classes = 10


model = dict(
    type='CoDETR',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # mean=[123.675, 116.28, 103.53],   # origin
        # std=[58.395, 57.12, 57.375],
        mean=[124.338, 118.218, 110.6955],
        std=[60.333, 58.956, 60.996],
        bgr_to_rgb=True,
        pad_mask=True,  # In instance segmentation, the mask needs to be padded
        pad_size_divisor=32),  # Padding the image to multiples of 32
    backbone=dict(
        type='SwinTransformer',
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    query_head=dict(
        type='CoDINOHead',
        num_classes=num_classes,
    ),
    roi_head=[
        dict(
            type='CoStandardRoIHead',
            bbox_roi_extractor=dict(
                type='SingleRoIExtractor',
                roi_layer=dict(
                    type='RoIAlign', output_size=7, sampling_ratio=0),
                out_channels=256,
                featmap_strides=[4, 8, 16, 32, 64],
                finest_scale=56),
            bbox_head=dict(
                type='Shared2FCBBoxHead',
                in_channels=256,
                fc_out_channels=1024,
                roi_feat_size=7,
                num_classes=num_classes,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.1, 0.1, 0.2, 0.2]),
                reg_class_agnostic=False,
                reg_decoded_bbox=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0 * num_dec_layer * loss_lambda),
                loss_bbox=dict(
                    type='GIoULoss',
                    loss_weight=10.0 * num_dec_layer * loss_lambda)))
    ],
    bbox_head=[
        dict(
            type='CoATSSHead',
            num_classes=num_classes,
            in_channels=256,
            stacked_convs=1,
            feat_channels=256,
            anchor_generator=dict(
                type='AnchorGenerator',
                ratios=[1.0],
                octave_base_scale=8,
                scales_per_octave=1,
                strides=[4, 8, 16, 32, 64, 128]),
            bbox_coder=dict(
                type='DeltaXYWHBBoxCoder',
                target_means=[.0, .0, .0, .0],
                target_stds=[0.1, 0.1, 0.2, 0.2]),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                gamma=2.0,
                alpha=0.25,
                loss_weight=1.0 * num_dec_layer * loss_lambda),
            loss_bbox=dict(
                type='GIoULoss',
                loss_weight=2.0 * num_dec_layer * loss_lambda),
            loss_centerness=dict(
                type='CrossEntropyLoss',
                use_sigmoid=True,
                loss_weight=1.0 * num_dec_layer * loss_lambda)),
    ])
    
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(512,512), (1024, 1024)],
                    keep_ratio=True)
            ],
            [
                dict(
                    type='RandomChoiceResize',
                    scales=[(400, 400), (500, 500), (600, 600)], # TODO
                    keep_ratio=True),
                dict(
                    type='RandomCrop',
                    crop_type='absolute_range',
                    crop_size=(384, 384),
                    allow_negative_crop=True),
                dict(
                    type='RandomChoiceResize',
                    scales=[(480, 480), (512, 512), (544, 544),  # TODO
                            (576, 576), (608, 608), (640, 640),
                            (672, 672), (704, 704), (736, 736),
                            (768, 768), (800, 800)],
                    keep_ratio=True)
            ]
        ]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
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
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
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
###
# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
            'sampling_offsets': dict(lr_mult=0.1),
            'reference_points': dict(lr_mult=0.1)
        }))
###
# learning policy
max_epochs = 30
train_cfg = dict(
    _delete_=True,
    type='EpochBasedTrainLoop',
    max_epochs=max_epochs,
    val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[11],
        gamma=0.1)
]

### Checkpoint ###
default_hooks = dict(
    checkpoint=dict(
        by_epoch=True, 
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

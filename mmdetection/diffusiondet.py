_base_ = [
    'projects/DiffusionDet/configs/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco.py'
]

# _base_ = [
#     'mmdet::_base_/datasets/coco_detection.py',
#     'mmdet::_base_/schedules/schedule_1x.py',
#     'mmdet::_base_/default_runtime.py'
# ]

### Setting ###
# dataset 설정을 해줍니다.
data_root='../../dataset/'
k='1'
batch_size = 4

custom_imports = dict(
    imports=['projects.DiffusionDet.diffusiondet'], allow_failed_imports=False)

# model settings
model = dict(
    type='DiffusionDet',
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        # mean=[123.675, 116.28, 103.53],   # origin
        # std=[58.395, 57.12, 57.375],
        mean=[124.338, 118.218, 110.6955], 
        std=[60.333, 58.956, 60.996],
        bgr_to_rgb=True,
        pad_size_divisor=32),
    bbox_head=dict(
        type='DynamicDiffusionDetHead',
        num_classes=10,
        feat_channels=256,
        num_proposals=500,
        num_heads=6,
        deep_supervision=True,
        prior_prob=0.01,
        snr_scale=2.0,
        sampling_timesteps=1,
        ddim_sampling_eta=1.0,
        criterion=dict(
            type='DiffusionDetCriterion',
            num_classes=10,
            assigner=dict(
                type='DiffusionDetMatcher',
                match_costs=[
                    dict(
                        type='FocalLossCost',
                        alpha=0.25,
                        gamma=2.0,
                        weight=2.0,
                        eps=1e-8),
                    dict(type='BBoxL1Cost', weight=5.0, box_format='xyxy'),
                    dict(type='IoUCost', iou_mode='giou', weight=2.0)
                ],
                center_radius=2.5,
                candidate_topk=5),
            loss_cls=dict(
                type='FocalLoss',
                use_sigmoid=True,
                alpha=0.25,
                gamma=2.0,
                reduction='sum',
                loss_weight=2.0),
            loss_bbox=dict(type='L1Loss', reduction='sum', loss_weight=5.0),
            loss_giou=dict(type='GIoULoss', reduction='sum',
                           loss_weight=2.0))),
    test_cfg=dict(
        use_nms=True,
        score_thr=0.5,
        min_bbox_size=0,
        nms=dict(type='nms', iou_threshold=0.5),
    ))

backend = 'pillow'
train_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.5),
    dict(
        type='RandomChoice',
        transforms=[[
            # dict(
            #     type='RandomChoiceResize',
            #     scales=[(480, 1333), (512, 1333), (544, 1333), (576, 1333), # TODO
            #             (608, 1333), (640, 1333), (672, 1333), (704, 1333),
            #             (736, 1333), (768, 1333), (800, 1333)],
            #     keep_ratio=True,
            #     backend=backend),
            dict(
                type='RandomChoiceResize',
                scales=[(512,512), (1024, 1024)],
                keep_ratio=True,
                backend=backend),
        ],
        [
            dict(
                type='RandomChoiceResize',
                # scales=[(400, 1333), (500, 1333), (600, 1333)], # TODO
                scales=[(400, 400), (500, 500), (600, 600)], # TODO
                keep_ratio=True,
                backend=backend),
            dict(
                type='RandomCrop',
                crop_type='absolute_range',
                crop_size=(384, 384),
                allow_negative_crop=True),
            dict(
                type='RandomChoiceResize',
                # scales=[(480, 1333), (512, 1333), (544, 1333),  # TODO
                #         (576, 1333), (608, 1333), (640, 1333),
                #         (672, 1333), (704, 1333), (736, 1333),
                #         (768, 1333), (800, 1333)],
                scales=[(480, 480), (512, 512), (544, 544),  # TODO
                        (576, 576), (608, 608), (640, 640),
                        (672, 672), (704, 704), (736, 736),
                        (768, 768), (800, 800)],
                keep_ratio=True,
                backend=backend)
        ]]),
    dict(type='PackDetInputs')
]

test_pipeline = [
    dict(
        type='LoadImageFromFile',
        backend_args=_base_.backend_args,
        imdecode_backend=backend),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True, backend=backend),   # TODO
    # If you don't have a gt annotation, delete the pipeline
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
    sampler=dict(type='InfiniteSampler'),   # TODO
    dataset=dict(
        filter_cfg=dict(filter_empty_gt=False, min_size=1e-5),
        pipeline=train_pipeline,
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'fold_{k}/stratified_train.json',
        data_prefix=dict(img='')))

val_dataloader = dict(
    batch_size=batch_size,
    dataset=dict(
        pipeline=test_pipeline,  
        data_root=data_root,
        metainfo=metainfo,
        ann_file=f'fold_{k}/stratified_val.json',
        data_prefix=dict(img='')))
test_dataloader = dict(
    batch_size=1,
    dataset=dict(
        pipeline=test_pipeline,  
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

# optimizer
optim_wrapper = dict(
    type='OptimWrapper',
    optimizer=dict(
        _delete_=True, type='AdamW', lr=0.000025, weight_decay=0.0001),
    clip_grad=dict(max_norm=1.0, norm_type=2))
train_cfg = dict(
    _delete_=True,
    type='IterBasedTrainLoop',
    # max_iters=450000,     # origin
    # val_interval=75000,
    max_iters=75000,
    val_interval=1000)

# learning rate
param_scheduler = [
    dict(
        type='LinearLR', start_factor=0.01, by_epoch=False, begin=0, end=1000),
    dict(
        type='MultiStepLR',
        begin=0,
        end=75000,
        by_epoch=False,
        milestones=[35000, 42000],
        gamma=0.1)
]

### Checkpoint ###
# evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP50')
log_processor = dict(by_epoch=False)
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

load_from = "https://download.openmmlab.com/mmdetection/v3.0/diffusiondet/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco/diffusiondet_r50_fpn_500-proposals_1-step_crop-ms-480-800-450k_coco_20230215_090925-7d6ed504.pth"
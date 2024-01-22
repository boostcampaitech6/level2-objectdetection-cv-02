_base_ = [
    '../_base_/models/cascade_rcnn_r50_fpn.py',
    '../common/mstrain_3x_coco.py',# '../_base_/default_runtime.py'
]

# # model settings - change IOU
# model = dict(
#     type='CascadeRCNN',
#     backbone=dict(
#         type='ResNet',
#         depth=50,
#         num_stages=4,
#         out_indices=(0, 1, 2, 3),
#         frozen_stages=1,
#         norm_cfg=dict(type='BN', requires_grad=True),
#         norm_eval=True,
#         style='pytorch',
#         init_cfg=dict(type='Pretrained', checkpoint='torchvision://resnet50')),
#     neck=dict(
#         type='FPN',
#         in_channels=[256, 512, 1024, 2048],
#         out_channels=256,
#         num_outs=5),
#     rpn_head=dict(
#         type='RPNHead',
#         in_channels=256,
#         feat_channels=256,
#         anchor_generator=dict(
#             type='AnchorGenerator',
#             scales=[8],
#             ratios=[0.5, 1.0, 2.0],
#             strides=[4, 8, 16, 32, 64]),
#         bbox_coder=dict(
#             type='DeltaXYWHBBoxCoder',
#             target_means=[.0, .0, .0, .0],
#             target_stds=[1.0, 1.0, 1.0, 1.0]),
#         loss_cls=dict(
#             type='CrossEntropyLoss', use_sigmoid=True, loss_weight=1.0),
#         loss_bbox=dict(type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)),
#     roi_head=dict(
#         type='CascadeRoIHead',
#         num_stages=3,
#         stage_loss_weights=[1, 0.5, 0.25],
#         bbox_roi_extractor=dict(
#             type='SingleRoIExtractor',
#             roi_layer=dict(type='RoIAlign', output_size=7, sampling_ratio=0),
#             out_channels=256,
#             featmap_strides=[4, 8, 16, 32]),
#         bbox_head=[
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=80,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.1, 0.1, 0.2, 0.2]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=80,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.05, 0.05, 0.1, 0.1]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
#                                loss_weight=1.0)),
#             dict(
#                 type='Shared2FCBBoxHead',
#                 in_channels=256,
#                 fc_out_channels=1024,
#                 roi_feat_size=7,
#                 num_classes=80,
#                 bbox_coder=dict(
#                     type='DeltaXYWHBBoxCoder',
#                     target_means=[0., 0., 0., 0.],
#                     target_stds=[0.033, 0.033, 0.067, 0.067]),
#                 reg_class_agnostic=True,
#                 loss_cls=dict(
#                     type='CrossEntropyLoss',
#                     use_sigmoid=False,
#                     loss_weight=1.0),
#                 loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
#         ]),
#     # model training and testing settings
#     train_cfg=dict(
#         rpn=dict(
#             assigner=dict(
#                 type='MaxIoUAssigner',
#                 pos_iou_thr=0.5,
#                 neg_iou_thr=0.3,
#                 min_pos_iou=0.3,
#                 match_low_quality=True,
#                 ignore_iof_thr=-1),
#             sampler=dict(
#                 type='RandomSampler',
#                 num=256,
#                 pos_fraction=0.5,
#                 neg_pos_ub=-1,
#                 add_gt_as_proposals=False),
#             allowed_border=0,
#             pos_weight=-1,
#             debug=False),
#         rpn_proposal=dict(
#             nms_pre=2000,
#             max_per_img=2000,
#             nms=dict(type='nms', iou_threshold=0.5),
#             min_bbox_size=0),
#         rcnn=[
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.5,
#                     neg_iou_thr=0.5,
#                     min_pos_iou=0.5,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.5,
#                     neg_iou_thr=0.5,
#                     min_pos_iou=0.5,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 pos_weight=-1,
#                 debug=False),
#             dict(
#                 assigner=dict(
#                     type='MaxIoUAssigner',
#                     pos_iou_thr=0.6,
#                     neg_iou_thr=0.7,
#                     min_pos_iou=0.7,
#                     match_low_quality=False,
#                     ignore_iof_thr=-1),
#                 sampler=dict(
#                     type='RandomSampler',
#                     num=512,
#                     pos_fraction=0.25,
#                     neg_pos_ub=-1,
#                     add_gt_as_proposals=True),
#                 pos_weight=-1,
#                 debug=False)
#         ]),
#     test_cfg=dict(
#         rpn=dict(
#             nms_pre=1000,
#             max_per_img=1000,
#             nms=dict(type='nms', iou_threshold=0.4),
#             min_bbox_size=0),
#         rcnn=dict(
#             score_thr=0.05,
#             nms=dict(type='nms', iou_threshold=0.5),
#             max_per_img=100)))

pretrained = 'https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_tiny_patch4_window7_224.pth'  # noqa
model = dict(
    # type='MaskRCNN', ## 모델은 cascade_rcnn을 사용하기 때문에 주석처리
    backbone=dict(
        _delete_=True,  ## 기존에 백본을 Resnet을 썼는데 Swin으로 쓰겠다. lr과 같은 다른 config에도 같이 사용이 가능한 인자
        type='SwinTransformer',
        embed_dims=96,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        window_size=7,
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.2,
        patch_norm=True,
        out_indices=(0, 1, 2, 3),
        with_cp=False,
        convert_weights=True,
        init_cfg=dict(type='Pretrained', checkpoint=pretrained)),
    neck=dict(in_channels=[96, 192, 384, 768])
        # [256, 512, 1024, 2048] = 기존의 in_channels . backcone마다 channel이 다르다.    
)

#change optimizer
#delete optimizer from mstrain_3x_coco.py file before running
optimizer = dict(
    _delete_=True,
    type='AdamW',
    lr=0.0004,
    betas=(0.9, 0.999),
    weight_decay=0.05,
    paramwise_cfg=dict(
        custom_keys={
            'absolute_pos_embed': dict(decay_mult=0.),
            'relative_position_bias_table': dict(decay_mult=0.),
            'norm': dict(decay_mult=0.)
        }))
optimizer_config = dict(grad_clip=None)
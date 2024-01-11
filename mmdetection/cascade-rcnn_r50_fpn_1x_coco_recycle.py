# 사용하고 싶은 model의 config 파일을 불러옵니다.
_base_ = '../mmdetection/configs/cascade_rcnn/cascade-rcnn_r50_fpn_1x_coco.py'

# model 끝단에 num_classes 부분을 바꿔주기 위해 해당 모듈을 불러와 선언해줍니다.
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10,
                reg_class_agnostic=True,    # 회귀분석이 클래스와 무관한가? 아니지 않나?
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10,
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0,
                               loss_weight=1.0)),
            dict(
                type='Shared2FCBBoxHead',
                num_classes=10,
                bbox_coder=dict(
                    type='DeltaXYWHBBoxCoder',
                    target_means=[0., 0., 0., 0.],
                    target_stds=[0.033, 0.033, 0.067, 0.067]),
                reg_class_agnostic=True,
                loss_cls=dict(
                    type='CrossEntropyLoss',
                    use_sigmoid=False,
                    loss_weight=1.0),
                loss_bbox=dict(type='SmoothL1Loss', beta=1.0, loss_weight=1.0))
        ]),
    train_cfg=dict(
        rcnn=[
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.5,
                    neg_iou_thr=0.5,
                    min_pos_iou=0.5,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.6,
                    neg_iou_thr=0.6,
                    min_pos_iou=0.6,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False),
            dict(
                assigner=dict(
                    type='MaxIoUAssigner',
                    pos_iou_thr=0.7,
                    neg_iou_thr=0.7,
                    min_pos_iou=0.7,
                    match_low_quality=False,
                    ignore_iof_thr=-1),
                sampler=dict(
                    type='RandomSampler',
                    num=512,
                    pos_fraction=0.25,
                    neg_pos_ub=-1,
                    add_gt_as_proposals=True),
                pos_weight=-1,
                debug=False)
        ]))

# custom hooks을 만들었다면 여기서 선언해 사용할 수 있습니다.
# /data/ephemeral/.vm/lib/python3.10/site-packages/mmdet/engine/hooks/
custom_hooks = [
    dict(type='SubmissionHook', test_out_dir='submit'),
    dict(type='MMDetWandbHook'),
]

# dataset 설정을 해줍니다.
data_root='../../dataset/'
k='0'
metainfo = {
    'classes': ('General trash', 'Paper', 'Paper pack', 'Metal', 'Glass',
                'Plastic', 'Styrofoam', 'Plastic bag', 'Battery', 'Clothing',),
    'palette': [
        (220, 20, 60), (119, 11, 32), (0, 0, 230), (106, 0, 228), (60, 20, 220),
        (0, 80, 100), (0, 0, 70), (50, 0, 192), (250, 170, 30), (255, 0, 0)
    ]
}
train_dataloader = dict(
    batch_size=8,
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

# 사용하는 모델의 pre-trainede된 checkpoint 경로/링크를 불러옵니다.
load_from = 'https://download.openmmlab.com/mmdetection/v2.0/cascade_rcnn/cascade_rcnn_r50_fpn_1x_coco/cascade_rcnn_r50_fpn_1x_coco_20200316-3dc56deb.pth'
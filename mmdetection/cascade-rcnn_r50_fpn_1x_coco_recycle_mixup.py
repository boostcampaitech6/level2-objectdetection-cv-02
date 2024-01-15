_base_ = [
    'cascade-rcnn_r50_fpn_1x_coco_recycle.py'
]

### Setting ###
data_root='../../dataset/'
k='1'
epoch = 10
batch_size = 16
resize = (1024,1024)

### Backbone ###
model = dict(
    roi_head=dict(),
    train_cfg=dict(),
)

### Dataset ###
train_pipeline = [  # NOTE: 배열은 전체 다 선언해주어야 합니다.
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=resize, keep_ratio=True),    # 사전 학습 모델의 scale을 그대로 사용할까요?
    dict(type='RandomFlip', prob=0.5),
    dict(type='MixUp',
        img_scale=resize),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=resize, keep_ratio=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'))
]

### Scheduler ###
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=epoch, val_interval=1)
auto_scale_lr = dict(enable=False, base_batch_size=16)


train_dataloader = dict(  
    batch_size=batch_size,
    dataset=dict(pipeline=train_pipeline, ann_file=f'fold_{k}/stratified_train.json'))
val_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, ann_file=f'fold_{k}/stratified_val.json'))
test_dataloader = dict(
    dataset=dict(pipeline=test_pipeline, ann_file=f'test.json'))

val_evaluator = dict(
    ann_file=data_root + f'fold_{k}/stratified_val.json')

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
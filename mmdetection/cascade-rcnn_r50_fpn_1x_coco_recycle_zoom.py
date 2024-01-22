### How to excute ###
# python tools/train.py config.py --cfg-options randomness.seed=2024

# 사용하고 싶은 model의 config 파일을 불러옵니다.
_base_ = [
    'cascade-rcnn_r50_fpn_1x_coco_recycle.py'
]

### Setting ###
# dataset 설정을 해줍니다. NOTE: 여기만 바꾸는 것이 아니라 우리가 선언한 변수들을 활용하는 부분까지 아래에 작성해주어야 합니다.
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
    dict(
        type='RandomChoiceResize',
        scales=[(768, 768), (896, 896), (1024, 1024)], # TODO
        keep_ratio=True),
    dict(
        type='RandomCrop',
        crop_type='absolute_range',
        crop_size=(700, 700),
        allow_negative_crop=True),
    dict(
        type='RandomChoiceResize',
        scales=[(768, 768), (832, 832), (896, 896),
                (960, 960), (1024, 1024)],
        keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
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

tta_model = dict(
    type='DetTTAModel',
    tta_cfg=dict(nms=dict(
                   type='nms',
                   iou_threshold=0.5),
                   max_per_img=100))

tta_pipeline = [
    dict(type='LoadImageFromFile',
        backend_args=None),
    dict(
        type='TestTimeAug',
        transforms=[[
            dict(type='Resize', scale=(1024, 1024), keep_ratio=True)
        ], [ # It uses 2 flipping transformations (flipping and not flipping).
            dict(type='RandomFlip', prob=1.),
            dict(type='RandomFlip', prob=0.)
        ], [
            dict(
               type='PackDetInputs',
               meta_keys=('img_id', 'img_path', 'ori_shape',
                       'img_shape', 'scale_factor', 'flip',
                       'flip_direction'))
       ]])]


default_hooks = dict(checkpoint=dict(interval=-1))  # no check-point, save only best result
evaluation = dict(interval=1, metric='bbox', save_best='bbox_mAP50')

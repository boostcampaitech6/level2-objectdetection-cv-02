### How to excute ###
# python tools/train.py config.py --cfg-options randomness.seed=2024

# 사용하고 싶은 model의 config 파일을 불러옵니다.
_base_ = [
    'cascade-rcnn_r50_fpn_1x_coco_recycle.py'
]

### Setting ###
# dataset 설정을 해줍니다.
data_root='../../dataset/'
k='1'
epoch = 10
batch_size = 16
resize = (1024,1024)
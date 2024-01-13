for k in 3 4
do
python tools/train.py \
    cascade-rcnn_r50_fpn_1x_coco_recycle_fold${k}_resize1024.py \
    --cfg-options randomness.seed=2024
done
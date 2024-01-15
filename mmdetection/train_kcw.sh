# for k in 1
# do
# python tools/train.py \
#     cascade-rcnn_r50_fpn_1x_coco_recycle_fold${k}_resize1024.py \
#     --cfg-options randomness.seed=2024
# done

# python tools/train.py \
#     cascade-rcnn_r50_fpn_1x_coco_recycle_centercrop.py \
#     --cfg-options randomness.seed=2024

# python tools/train.py \
#     cascade-rcnn_r50_fpn_1x_coco_recycle_classaware.py \
#     --cfg-options randomness.seed=2024

# python tools/train.py \
#     efficientdet.py \
#     --cfg-options randomness.seed=2024

python tools/train.py \
    diffusiondet.py \
    --cfg-options randomness.seed=2024

# python tools/train.py \
#     co-detr.py \
#     --cfg-options randomness.seed=2024

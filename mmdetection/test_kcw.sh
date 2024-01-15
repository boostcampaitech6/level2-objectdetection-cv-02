# for k in 3 4
# do
# python tools/test.py \
#     cascade-rcnn_r50_fpn_1x_coco_recycle_fold${k}_resize1024.py \
#     work_dirs/cascade-rcnn_r50_fpn_1x_coco_recycle_fold${k}_resize1024/epoch_10.pth
# done

python tools/test.py \
    diffusiondet.py \
    work_dirs/diffusiondet/"best_coco_General trash_precision_iter_37000.pth"

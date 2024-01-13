for k in 3 4
do
python tools/test.py \
    cascade-rcnn_r50_fpn_1x_coco_recycle_fold${k}_resize1024.py \
    work_dirs/cascade-rcnn_r50_fpn_1x_coco_recycle_fold${k}_resize1024/epoch_10.pth
done

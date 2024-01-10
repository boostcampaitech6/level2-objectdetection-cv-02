import argparse
from mmcv import Config
from mmdet.datasets import build_dataset
from mmdet.models import build_detector
from mmdet.apis import train_detector
from mmdet.datasets import build_dataloader, build_dataset, replace_ImageToTensor
from mmdet.utils import get_device


def init(args):
    classes = ("General trash", "Paper", "Paper pack", "Metal", "Glass", 
           "Plastic", "Styrofoam", "Plastic bag", "Battery", "Clothing")
    
    # config file 들고오기
    cfg = Config.fromfile(args.config_path)

    root = args.data_dir
    resize = args.resize

    # dataset config 수정
    cfg.data.train.classes = classes
    cfg.data.train.img_prefix = root
    cfg.data.train.ann_file = root + 'train.json' # train json 정보
    cfg.data.train.pipeline[2]['img_scale'] = resize # Resize

    cfg.data.test.classes = classes
    cfg.data.test.img_prefix = root
    cfg.data.test.ann_file = root + 'test.json' # test json 정보
    cfg.data.test.pipeline[1]['img_scale'] = resize # Resize

    cfg.data.samples_per_gpu = args.samples_per_gpu

    cfg.seed = args.seed
    cfg.gpu_ids = [0]
    cfg.work_dir = args.model_dir

    cfg.model.roi_head.bbox_head.num_classes = 10

    cfg.optimizer_config.grad_clip = dict(max_norm=35, norm_type=2)
    cfg.checkpoint_config = dict(max_keep_ckpts=3, interval=1)
    cfg.device = get_device()
    
    return cfg


def get_datasets(cfg):
    return [build_dataset(cfg.data.train)]


def main(args):
    cfg = init(args)
    datasets = get_datasets(cfg)
    model = build_detector(cfg.model)
    model.init_weights()
    train_detector(model, datasets[0], cfg, distributed=False, validate=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument(
        "--config_path", type=str, default='./configs/faster_rcnn/faster_rcnn_vit_fpn_1x_coco.py'
    )
    parser.add_argument(
        "--data_dir", type=str, default='../../dataset/'
    )
    parser.add_argument(
        "--model_dir", type=str, default='./work_dirs/faster_rcnn_vit_fpn_1x_trash'
    )
    parser.add_argument(
        "--resize", type=int, default=[512,512]
    )
    parser.add_argument(
        "--samples_per_gpu", type=int, default=4
    )
    parser.add_argument(
        "--seed", type=int, default=0
    )
    
    args = parser.parse_args()
    print(args)

    main(args)
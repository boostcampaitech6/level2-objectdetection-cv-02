# How to use MMDetection(v2)

You have to checkout your branch location to `mmdetection`

Pull `mmdetection` branch from GitHub

## Train

1. Use train.py<br>
    MMDetection provides a variety of config files. You must modify the config.py of model to option you want to use. If you want a executable config file, you can use `mmdetection/configs/atss/swin-l.py`.
    ```
    python mmdetection/tools/train.py [config.py]
    ```

    
2. Use ipynb
    ```
    jupyter notebook --allow-root
    ```
    - Open `faster_rcnn_train.ipynb` on jupyter notebook and execute cells sequentially


## Inference
After training, There is a model.pth that you trained in `/work_dirs` directory. You can use model.pth at testing time to check the performance of the model.

1. Use test.py
    ```
    python mmdetection/tools/test.py [config.py] [model.pth] [eval option]
    ```

2. Use ipynb
    ```
    jupyter notebook --allow-root
    ```
    - Open `faster_rcnn_inference.ipynb` on jupter notebook and execute cells sequentially



## Connect your own Wandb

There is code about Weight&Biases in both of `train.py` and `faster_rcnn_train.ipynb`. To connect your own Wandb, You must modify the code below.

```
log_config = dict(
    interval=100,
    hooks=[
        dict(type='TextLoggerHook', interval=500),
        dict(type='WandbLoggerHook',interval=100,
            init_kwargs=dict(
                project='project-name',
                entity = 'team-name',
                name = 'run-name'),
            ),
    ])
```

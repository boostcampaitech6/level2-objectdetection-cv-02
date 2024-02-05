# How to use Yolov8
yolov8
|
|-- pseudo_labeling.py
|-- readme_pseudo_labeling.md
|-- requirements.txt
|-- stratified
    |-- coco_to_yolo.py
    |-- stratified.py
    `-- supervisely_to_original.py
|-- inference.py
`-- train.py

## Create Dataset

```
cd stratified
```

If you use supervisely

```
python supervisely_to_original.py --root path/to/your_dataset_dir --inputs ['/instances.json/'] --output change.json
```

1. COCO Dataset to YOLO Dataset

```
python coco_to_yolo.py -j path/to/your_json_file -o path/to/output_folder
```

2. Stratified

you need to edit dataset path in stratified.py

If you want to classify images and labels, please uncomment them

annotation path
annotation = '../../dataset/train.json'```


create directory path
```
os.makedirs(f'path_to/fold_{fold}/train', exist_ok=True)
os.makedirs(f'path_to/fold_{fold}/val', exist_ok=True)
```

copy image path
```
shutil.copy(f'path_to_original_path/{path}', f'path/to/new_dataset_path/fold_{fold}/train/{filename}')
```
```
shutil.copy(f'path_to_original_path/{path}', f'path/to/new_dataset_path/fold_{fold}/val/{filename}')
```

copy label path
```
src = f'path/to/original_train_path/{filename}'
dst = f'path/to/new_dataset_path/fold_{fold}/train/{filename}'
```
```
src = f'path/to/original_train_path/{filename}'
dst = f'path/to/new_dataset_path/fold_{fold}/val/{filename}'
```


## pseudo_labeling

if you want to pseudo labeling, you need to edit path

more information in readme_pseudo_labeling.md

pseudo_labeling.py
```
model = YOLO("path/to/your_trained_model_path/weight.pt")
```
```
predictions = model.predict("path/to/no_label_dataset")
```
```
image_path = f"path/to/no_label_dataset/{i:04}.jpg"
```

```
python pseudo_labeling.py
```
## training

1. you need to download model(.pt) file from "https://github.com/ultralytics/ultralytics" and put it in yolov8 folder.

2. you can edit train.py file to change hyperparameter.

```
# train.py

from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.train(data="./ultralytics/data/data.yaml",
            epochs=200, 
            batch=4,
            imgsz=1024, 
            project="yolo_train", 
            name="yolov8x", 
            device=0,
            multi_scale = True,
            lr0 = 0.00001,
            lrf = 0.01,
            optimizer = "AdamW",
            )
```

3. run train.py

```
python train.py
```

## inference

1. you need to edit check_point path in inference.py file.

```
# inference.py

check_point = './yolo_train/yolov8x12/weights/best.pt'
```

2. run inference.py

```
python inference.py
```

## Result
#### Yolov8x with k-fold ensemble (fold 0~4)
|Local mAP|final mAP|
|:---:|:---:|
|0.6104|0.5863|
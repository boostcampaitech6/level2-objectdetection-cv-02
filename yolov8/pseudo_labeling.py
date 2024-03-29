from ultralytics import YOLO
import os
model = YOLO("/data/ephemeral/home/yolov8/yolo_train/yolov8l_batch=8, imgsz=1024_augment=True,mosaic = 0.5,mixup=0.3/weights/best.pt")


# 훈련된 모델을 사용하여 레이블이 없는 데이터에 대한 예측을 생성
predictions = model.predict("/data/ephemeral/home/dataset/pseudo_labeling/train")
# 예측을 레이블로 사용하여 레이블이 없는 데이터를 레이블이 있는 데이터로 변환.
for i, result in enumerate(predictions):
    image_path = f"/data/ephemeral/home/dataset/pseudo_labeling/train/{i:04}.jpg"
    label_path = os.path.splitext(image_path)[0] + ".txt"
    with open(label_path, "w") as f:
        for box, label in zip(result.boxes.xywhn, result.boxes.cls):
            x_center, y_center, width, height = box
            f.write(f"{int(label)} {x_center} {y_center} {width} {height}\n")

# model.train(data='/data/ephemeral/home/yolov8/ultralytics/cfg/datasets/pseudo_labeling.yaml',
#              epochs=30, 
#              batch=16,
#              imgsz=700, 
#              multi_scale = True,
#              lr0 = 0.00001,
#              lrf = 0.00001,
#              optimizer = 'AdamW',
#              project="yolo_train", 
#              name="yolov8m_fold43_img800_lr0,lrf(0.00001)_epoch2002_pseudo_labeling", 
#              device=0)
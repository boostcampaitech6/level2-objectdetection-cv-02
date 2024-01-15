from ultralytics import YOLO

model = YOLO("yolov8m.pt")
model.train(data="/data/ephemeral/home/yolov8/ultralytics/cfg/datasets/data.yaml",
             epochs=200, 
             #cfg='/data/ephemeral/home/yolov8/ultralytics/cfg/models/v8/yolov8.yaml',
             batch=16,
             imgsz=800, 
             multi_scale = True,
             lr0 = 0.00001,
             lrf = 0.00001,
             mosaic= 0.5,
             mixup= 0.7,
             optimizer = 'AdamW',
             project="yolo_train", 
             name="yolov8m_AdamW_mosaic:0.5, mixup: 0.7", 
             device=0)
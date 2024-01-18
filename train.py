from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.train(data="/data/ephemeral/home/yolov8/ultralytics/cfg/datasets/data.yaml",
             epochs=200, 
             #cfg='/data/ephemeral/home/yolov8/ultralytics/cfg/models/v8/yolov8.yaml',
             batch=4,
             imgsz=1024, 
             multi_scale = True,
             lr0 = 0.00001,
             lrf = 0.01,
             optimizer = 'AdamW',
             project="yolo_train", 
             name="yolov8x_fold2", 
             device=0)
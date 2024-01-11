from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.train(data="/data/ephemeral/home/yolov8/ultralytics/cfg/datasets/data.yaml",
             epochs=100, 
             #cfg='/data/ephemeral/home/yolov8/ultralytics/cfg/models/v8/yolov8.yaml',
             batch=16,
             imgsz=640, 
             multi_scale = True,
             project="yolo_train", 
             name="yolov8x", 
             device=0)
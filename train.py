from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="/data/ephemeral/home/yolov8/ultralytics/data/data.yaml",
             epochs=100, 
             #cfg='/data/ephemeral/home/yolov8/ultralytics/cfg/models/v8/yolov8.yaml',
             batch=64,
             imgsz=1024, 
             project="yolo_train", 
             name="yolov8n", 
             device=0)
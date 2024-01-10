from ultralytics import YOLO

model = YOLO("yolov8n.pt")
model.train(data="/data/ephemeral/home/yolov8/ultralytics/data/data.yaml", epochs=70, imgsz=640, project="yolo_train", name="yolov8n", device=0)
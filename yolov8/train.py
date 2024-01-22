from ultralytics import YOLO

model = YOLO("yolov8x.pt")
model.train(data="./ultralytics/data/data.yaml",
            epochs=200, 
            #cfg='/data/ephemeral/home/yolov8/ultralytics/cfg/models/v8/yolov8.yaml',
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
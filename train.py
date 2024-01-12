from ultralytics import YOLO

model = YOLO("/data/ephemeral/home/yolov8/yolo_train/yolov8m_fold43/weights/best.pt")
model.train(data="/data/ephemeral/home/yolov8/ultralytics/cfg/datasets/data.yaml",
             epochs=200, 
             #cfg='/data/ephemeral/home/yolov8/ultralytics/cfg/models/v8/yolov8.yaml',
             batch=16,
             imgsz=800, 
             multi_scale = True,
             lr0 = 0.00001,
             lrf = 0.00001,
             optimizer = 'AdamW',
             project="yolo_train", 
             name="yolov8m_fold43_retrain(lr0,lrf(0.00001)_epoch2002, AdamW)", 
             device=0)

# # 2. 훈련된 모델을 사용하여 레이블이 없는 데이터에 대한 예측을 생성합니다.
# predictions = model.predict("/path/to/unlabeled/data")

# # 3. 이 예측을 레이블로 사용하여 레이블이 없는 데이터를 레이블이 있는 데이터로 변환합니다.
# pseudo_labeled_data = "/path/to/unlabeled/data"
# pseudo_labeled_data.labels = predictions
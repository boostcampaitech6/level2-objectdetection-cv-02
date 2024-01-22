### Dataset 구조

Dataset
|-- pseudo_labeling
|   |-- test
|   `-- val

### pseudo_labeling.yaml 경로
```
/data/ephemeral/home/yolov8/ultralytics/cfg/datasets/pseudo_labeling.yaml
```

### 진행전에 해야할 것
1. test디렉토리 복사 해서 pseudo_labeling 디렉토리를 새로 만들 것을 추천
2. stratified 한 fold 중에서 하나를 골라 val디렉토리를 복사 후 pseudo_labeling 디렉토리에 붙여넣기

### 진행

1. model = YOLO(' ')
 - 안에 우리가 직접 train한 모델 weight경로 넣기

2. predictions = model.predict(' ')
 - label이 없는 test데이터를 이용
 - test데이터 디렉토리를 그대로 복사해서 다른 곳에 넣는 것을 추천 (test데이터 디렉토리 안에 txt파일이 생성됨)
 - test데이터의 경로 넣기

3. image_path
 - 마찬가지로 test데이터의 경로 넣기

4. model.train(data= ' ' ~~)
 - data안에 pseudo_labeling.yaml 파일 넣기
 - pseudo_labeling.yaml안에는 기존 data.yaml과 비슷하게 train, val , class 정보가 담겨져 있음

### Train
```
python pseudo_labeling.py
```

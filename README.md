# [AI Tech 6th CV Course] 팀 도라에몽

## Table of content

- [Overview](#Overview)
- [Member](#Member)
- [Approach](#Approach)
- [Result](#Result)
- [File Tree](#filetree)
- [Usage](#Code)




<br></br>
## Overview <a id = 'Overview'></a>
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-01/assets/5775698/b55bddbb-6c44-47d1-ac2c-d290354bf903)

우리는 수많은 쓰레기를 배출하면서 지구의 환경파괴, 야생동물의 생계 위협 등 여러 문제를 겪고 있습니다. 분리수거는 이러한 환경 부담을 줄일 수 있는 방법 중 하나입니다. 잘 분리배출 된 쓰레기는 자원으로서 가치를 인정받아 재활용되지만, 잘못 분리배출 되면 그대로 폐기물로 분류되어 매립 또는 소각되기 때문입니다.

이러한 문제는 쓰레기를 줍는 드론, 쓰레기 배출 방지 비디오 감시, 인간의 쓰레기 분류를 돕는 AR 기술과 같은 여러 기술을 통해서 조금이나마 개선이 가능합니다. 따라서 사진에서 쓰레기를 Detection 하는 모델을 만들어 이러한 문제점을 해결해보고자 합니다. 문제 해결을 위해 일반 쓰레기, 플라스틱, 종이, 유리 등 10 종류의 쓰레기를 구별하는 모델을 개발하는 것이 프로젝트 목적입니다.

<br></br>
## Member <a id = 'Member'></a>

|김찬우|남현우|류경엽|이규섭|이현지|한주희|
|:--:|:--:|:--:|:--:|:--:|:--:|
|<a href='https://github.com/uowol'><img src='https://avatars.githubusercontent.com/u/20416616?v=4' width='100px'/></a>|<a href='https://github.com/nhw2417'><img src='https://avatars.githubusercontent.com/u/103584775?s=88&v=4' width='100px'/></a>|<a href='https://github.com/kylew1004'><img src='https://avatars.githubusercontent.com/u/5775698?s=88&v=4' width='100px'/></a>|<a href='https://github.com/9sub'><img src='https://avatars.githubusercontent.com/u/113101019?s=88&v=4' width='100px'/></a>|<a href='https://github.com/solee328'><img src='https://avatars.githubusercontent.com/u/22787039?s=88&v=4' width='100px'/></a>|<a href='https://github.com/jh7316'><img src='https://avatars.githubusercontent.com/u/95545960?s=88&v=4' width='100px'/></a>|


<br></br>

## Approach <a id = 'Approach'></a>
### Architecture
- Faster R-CNN
- Cascade R-CNN
- YOLO V8
- CO-DETR
- DiffusionDet
- ATSS
### Backbone model
- ResNet(TorchVision, Faster R-CNN)
- Swin Transformer(MMDetection)

<br></br>

## Result <a id = 'Result'></a>

### Scoreboard
![image](https://github.com/boostcampaitech6/level1-imageclassification-cv-01/assets/5775698/c2205704-9790-4930-ac47-a77db114114c)

### Wrap Reports
<a href="https://file.notion.so/f/f/a4e6899b-3c12-44ea-b34a-079e598135db/c12f8ce7-c049-4c7d-af8f-c9f9f382e0c6/Object_Det_CV_%ED%8C%80%EB%A6%AC%ED%8F%AC%ED%8A%B8(02%EC%A1%B0).pdf?id=483ded0f-080b-430f-896e-b488b0c5a26c&table=block&spaceId=a4e6899b-3c12-44ea-b34a-079e598135db&expirationTimestamp=1706054400000&signature=ADqJGdqpfiX8Dn8fwqnBJap_QN1XpOM8hUPw-VlMZUA&downloadName=Object+Det_CV_%ED%8C%80%EB%A6%AC%ED%8F%AC%ED%8A%B8%2802%EC%A1%B0%29.pdf" target="_blank">Object Detection Wrapup Reports</a>


<br></br>

## File Tree <a id = 'filetree'></a>
```
level2-objectdetection-cv-02
|
.
.
|
|── tools
|    |- annotation_analysis.py
|    |- clean_supervisely.py
|    |- stratified_group_k_fold.py
|    └─ submission_visualization.ipynb
|
|── mmdetection(v2)
|    |- configs
|    |- tools
|    .   |- train.py
|    .   └─ test.py
|    |- faster_rcnn_train.ipynb
|    └─ faster_rcnn_inference.ipynb
|
|── mmdetection(v3)
|    |- configs
|    .
|    .
|    └─ tools
|        |- train.py
|        └─ test.py
|
|── yolov8
|    |- train.py
|    |- inference.py
|    |- wbf_ensemble.ipynb
|    |- stratified
|    .  |- coco_to_yolo.py
|    .  |- stratified.py
|    .  └─ supervisely_to_original.py
|    └─ pseudo_labeling.py
|
└── README.md
```

<br></br>
## Usage <a id = 'Code'></a>

### Package install

### Model Train & Inference
- [Yolov8](./docs/Yolov8.md)
- [MMDetection(v2)](./docs/MMDetection(v2).md)
- [MMDetection(v3)](./docs/MMDetection(v3).md)

### Extra Tools
- [Tools](./docs/Tools.md)
    1. Annotation Analysis (EDA)
    2. Clean Supervisely
    3. Stratified Group K Fold
    4. Submission Visualization

### [WBF ensemble](./docs/wbf_ensemble.md)
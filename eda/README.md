## Data Preview
- Task: Object detection
- \# of train dataset : 4883
- \# of test dataset : 4871
- Classes : General trash, Paper, Paper pack, Metal, Glass, Plastic, Styrofoam, Plastic bag, Battery, Clothing
- Image size : 1024 x 1024

## Input Data
- Format : COCO
```
├── dataset
    ├── train.json  (annotation file)
    ├── test.json   (annotation file)
    ├── train       (image folder)
    └── test        (image folder)
```

### Annotation File Preview (.json)
- images
    - id: 파일 안에서 image 고유 id, ex) 1
    - height: 1024
    - width: 1024
    - filename: train/{XXXX}.jpg
- annotations
    - id: 파일 안에 annotation 고유 id, ex) 1
    - bbox: 객체가 존재하는 박스의 좌표 
    (xmin, ymin, w, h)
    - area: 객체가 존재하는 박스의 크기
    - category_id: 객체가 해당하는 class의 id
    - image_id: annotation이 표시된 이미지 고유 id
    - is_crowd: 단일 객체(0)인지 여러 객체의 집합(1)인지 나타내는 binary 값

## Ouput Data
- Format : Pascal VOC
```
image_id, PredictionString
(test/0000.jpg), (label, score, xmin, ymin, xmax, ymax)
```

## Todo
- [ ] ss
- [ ] 
- [ ] 

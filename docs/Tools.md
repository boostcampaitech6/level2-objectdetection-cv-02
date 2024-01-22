# How to use Tools
## Index
1. Annotation Analysis
2. Clean Supervisely
3. Stratified Group K Fold
4. Submission Visualization

## Annotation Analysis
### Overview
- COCO format의 json파일을 불러와 다양한 인사이트를 추출합니다.
    - (hist)이미지 별 annotation, 즉 boundary box의 개수의 분포
    - (hist)클래스 별 annotation, 즉 boundary box의 개수의 분포
    - (hist)이미지 별 class 개수의 분포
    - (heat)boundary box의 position 시각화
    - (box)boundary box의 부피의 분포
    - (box)boundary box의 가로 세로 비율의 분포
    - (heat)class 별 상관관계 시각화
    - (text)각 클래스별 bbox들의 개수의 통계치
    - (text)각 클래스별 bbox들의 area의 평균의 통계치
    - (text)각 클래스별 bbox들의 aspect_ratio의 평균의 통계치
    - (text)각 이미지별 bbox들의 area의 평균의 통계치
    - (text)각 이미지별 bbox들의 aspect_ratio의 평균의 통계치
    - (text)각 이미지별 class의 개수의 통계치

### Usage
- `[--anno_path]`의 파일을 읽어 `[--output_path]/eda`의 경로에 결과물들을 저장합니다. 
```bash
python tools/annotation_analysis.py \
    --anno_path path/to/annotation.json \
    --output_path ./output
```
## Clean Supervisely
### Thanks to 강승환_T6003
- [Reference Link](https://next.stages.ai/competitions/266/discussion/343/post/2462)
### Overview

- Supervisely에서 리라벨링한 데이터를 다운받으면 원래의 데이터와 다음의 내용이 달라집니다.
    - category_id가 1씩 증가합니다. (원래 : 0 ~ (N-1) , 수정 후 : 1 ~ N)
    - image_id가 전부 증가합니다. (원래 : 1, 2 ~ ... , 수정 후 : K, K+1, ...)
- 기존 코드에서 문제가 발생할 수 있기 때문에 본 스크립트는 수정 후 데이터를 원래의 형태로 변환해줍니다.

### Usage
```bash
python tools/supervisly_to_coco.py \
    --root {root dir of json files} \
    --inputs [\'{name of json file from Supervisely}\'] \
    --range {json file에 맞는 범위들} \
    --output {output json 파일 이름}
```

### Example
```sh
python tools/supervisly_to_coco.py \
    --root {root dir of json files} \
    --inputs {[\'instances.json\']} \
    --range [0,4882] \
    --output clean.json
```

## Stratified Group K Fold
### Thanks to 이수아_조교
- [Reference Link](https://next.stages.ai/competitions/266/discussion/343/post/2373)

### Overview
- 클래스 불균형의 문제로 완전 랜덤 추출을 사용할 경우 원래의 데이터 분포와 다른 분포의 샘플이 얻어질 수 있습니다.
- 따라서 원래 데이터의 분포를 유지하면서 샘플링하는, Stratified Group K Fold 방법을 사용하면 K개의 Fold를 나누어 Train/Validation Set을 구할 수 있습니다.

### Usage
- 기본으로 k=5, shuffle=True, random_state=411가 주어집니다. 수정을 원하신다면 `tools/stratified_group_k_fold.py`의 24번째 줄을 수정하세요.
```sh
python tools/stratified_group_k_fold.py
```


## Submission Visualization
### Overview
- 학습한 모델이 테스트 데이터에 대해 예측한 결과인 `submission.csv` 파일을 읽고 예측한 boundary boxes를 시각화합니다.
- 모델이 잘 예측했는지, 해당 모델이 갖는 특징은 무엇인지 캐치할 수 있습니다. 
- 내용을 입력하여 관심있는 이미지를 따로 저장할 수도 있습니다.  

### Usage
- `tools/submission_visualization.ipynb`를 실행합니다.
- 이미지가 한 장씩 출력되는데, `q`를 입력하여 종료하거나 원하는 텍스트를 입력하여 `output/submission` 폴더에 해당 텍스트를 이름으로 한 이미지로 저장할 수 있습니다.
- `start_num`을 수정하여 원하는 index의 이미지부터 시각화할 수 있습니다.
import json
import numpy as np
import os
import shutil
from sklearn.model_selection import StratifiedGroupKFold

annotation = '../../dataset/train.json'
# load json annotation
with open(annotation) as f: 
    data = json.load(f)

var = [(ann['image_id'], ann['category_id']) for ann in data['annotations']]
X = np.ones((len(data['annotations']),1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

# 이미지 ID와 경로 생성
id_to_path = {img['id']: img['file_name'] for img in data['images']}

for fold, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
    train_ids, val_ids = groups[train_idx], groups[val_idx]
    train_labels, val_labels = y[train_idx], y[val_idx]
    
    # 필요한 디렉토리 생성
    os.makedirs(f'../../dataset/stratified_newlabel/fold_{fold}/train', exist_ok=True)
    os.makedirs(f'../../dataset/stratified_newlabel/fold_{fold}/val', exist_ok=True)

    #labels와 img를 분류하고싶다면 아래 주석을 해제
    #os.makedirs(f'../../dataset/stratified_newlabel/fold_{fold}/img/train', exist_ok=True)
    #os.makedirs(f'../../dataset/stratified_newlabel/fold_{fold}/img/val', exist_ok=True)
    #os.makedirs(f'../../dataset/stratified/fold_{fold}/label/train', exist_ok=True)
    #os.makedirs(f'../../dataset/stratified/fold_{fold}/label/val', exist_ok=True)
     # 훈련 이미지 파일 복사
    for id in train_ids:
        path = id_to_path[id]
        filename = os.path.basename(path)
        shutil.copy(f'../../dataset/{path}', f'../../dataset/stratified_newlabel/fold_{fold}/train/{filename}')

        #labels와 img를 분류하고싶다면 아래 주석을 해제
        #shutil.copy(f'../../dataset/{path}', f'../../dataset/stratified_newlabel/fold_{fold}/img/train/{filename}')

    # 검증 이미지 파일 복사
    for id in val_ids:
        path = id_to_path[id]
        filename = os.path.basename(path)
        shutil.copy(f'../../dataset/{path}', f'../../dataset/stratified_newlabel/fold_{fold}/val/{filename}')

    # 훈련 레이블 복사
    for id in train_ids:
        filename = f'{id:04}.txt'
        print(filename)
        src = f'../../dataset/train/{filename}'
        dst = f'../../dataset/stratified_newlabel/fold_{fold}/train/{filename}'

        #labels와 img를 분류하고싶다면 아래 주석을 해제
        #dst = f'../../dataset/stratified/fold_{fold}/label/train/{filename}'

        shutil.copy(src, dst)

    # 검증 레이블 복사
    for id in val_ids:
        filename = f'{id:04}.txt'
        src = f'../../dataset/train/{filename}'
        dst = f'../../dataset/stratified_newlabel/fold_{fold}/val/{filename}'
        
        #labels와 img를 분류하고싶다면 아래 주석을 해제
        #dst = f'../../dataset/stratified/fold_{fold}/label/val/{filename}'

        shutil.copy(src, dst)
import json
import numpy as np
import pandas as pd 
import os
import shutil
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
from copy import deepcopy

annotation_path = '../../dataset/train.json'
output_dir = './output/stratified'

# load json annotation
with open(annotation_path) as f: ori_train = json.load(f)

# print(ori_train['images'][0].keys()); raise


var = [(ann['image_id'], ann['category_id']) for ann in ori_train['annotations']]
X = np.ones((len(ori_train['annotations']),1))
y = np.array([v[1] for v in var])
groups = np.array([v[0] for v in var])

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=411)

def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]
    
distrs = [get_distribution(y)]
index = ['training set']

for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X, y, groups)):
    train_y, val_y = y[train_idx], y[val_idx]               # category_id
    train_gr, val_gr = groups[train_idx], groups[val_idx]   # image_id

    assert len(set(train_gr) & set(val_gr)) == 0    # 동일한 image를 포함하면 error 
    
    distrs.append(get_distribution(train_y))
    distrs.append(get_distribution(val_y))
    index.append(f'train - fold{fold_ind}')
    index.append(f'val - fold{fold_ind}')

    # 필요한 디렉토리 생성
    os.makedirs(f'{output_dir}/fold_{fold_ind}/', exist_ok=True)

    # annotations 파일 생성
    filename = os.path.basename(annotation_path)
    stratified_train = deepcopy(ori_train)
    stratified_train['images'] = [x for x in stratified_train['images'] if x['id'] in train_gr]
    stratified_train['annotations'] = [x for x in stratified_train['annotations'] if x['image_id'] in train_gr]
    with open(f'{output_dir}/fold_{fold_ind}/stratified_train.json', 'w') as f: 
        json.dump(stratified_train, f)
    stratified_val = deepcopy(ori_train)
    stratified_val['images'] = [x for x in stratified_val['images'] if x['id'] in val_gr]
    stratified_val['annotations'] = [x for x in stratified_val['annotations'] if x['image_id'] in val_gr]
    with open(f'{output_dir}/fold_{fold_ind}/stratified_val.json', 'w') as f: 
        json.dump(stratified_val, f)

categories = [d['name'] for d in ori_train['categories']]
df = pd.DataFrame(distrs, index=index, columns = [categories[i] for i in range(np.max(y)+1)])
df.to_csv(f'{output_dir}/fold_distribution.csv', index_label='domain')

print(pd.read_csv(f'{output_dir}/fold_distribution.csv'))
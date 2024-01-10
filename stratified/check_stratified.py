import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from collections import Counter
import pandas as pd

# X, y, groups, cv, data 정의 필요

def get_distribution(y):
    y_distr = Counter(y)
    y_vals_sum = sum(y_distr.values())

    return [f'{y_distr[i]/y_vals_sum:.2%}' for i in range(np.max(y) +1)]

distrs = [get_distribution(y)]
index = ['training set']

for fold_ind, (train_idx, val_idx) in enumerate(cv.split(X,y, groups)):
    train_y, val_y = y[train_idx], y[val_idx]
    distrs.append(get_distribution(train_y))
    distrs.append(get_distribution(val_y))
    index.append(f'train - fold{fold_ind}')
    index.append(f'val - fold{fold_ind}')

categories = [d['name'] for d in data['categories']]
pd.DataFrame(distrs, index=index, columns = [categories[i] for i in range(np.max(y) + 1)])
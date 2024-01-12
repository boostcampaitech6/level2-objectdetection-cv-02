import cv2
import os 
import albumentations as A 
from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
import pandas as pd 
import torch 
from tqdm import tqdm

data_path = "/data/ephemeral/home/dataset/train/"
image_size = 1024
batch_size = 64
num_workers = 0

augs = A.Compose(
    [
        A.Resize(height=image_size, width=image_size),
        A.Normalize(mean=(0, 0, 0), std=(1, 1, 1)),
        ToTensorV2(),
    ]
)

# placeholders
psum = torch.tensor([0.0, 0.0, 0.0])
psum_sq = torch.tensor([0.0, 0.0, 0.0])

for image in tqdm(data_list:=os.listdir(data_path)[:1000]):
    path = os.path.join(data_path, image)
    image = cv2.imread(path, cv2.COLOR_BGR2RGB)
    if augs is not None:
        image = augs(image=image)["image"]
    # print(image.shape)
    psum += image.sum(axis=[1, 2])
    psum_sq += (image**2).sum(axis=[1, 2])
    
count = len(data_list) * image_size * image_size

# mean and std
total_mean = psum / count
total_var = (psum_sq / count) - (total_mean**2)
total_std = torch.sqrt(total_var)

# output
print("mean: " + str(total_mean))
print("std:  " + str(total_std))


# display images
# for batch_idx, inputs in enumerate(image_loader):
#     fig = plt.figure(figsize=(14, 7))
#     for i in range(8):
#         ax = fig.add_subplot(2, 4, i + 1, xticks=[], yticks=[])
#         plt.imshow(inputs[i].numpy().transpose(1, 2, 0))
#     break
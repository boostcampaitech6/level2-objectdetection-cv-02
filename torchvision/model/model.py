import torch.nn as nn
import torch.nn.functional as F
from base.base_model import BaseModel

import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


class fasterrcnn_resnet50_fpn(BaseModel):
    def __init__(self, num_classes):
        super().__init__()
        self.model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        in_features = self.model.roi_heads.box_predictor.cls_score.in_features
        self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # https://github.com/pytorch/vision/blob/main/torchvision/models/detection/roi_heads.py
        # classification loss = F.cross_entropy
        # box_loss = F.smooth_l1_loss
        # self.model.roi_heads.fastrcnn_loss = custom_loss

    def inference(self, image):
        return self.model(image)

    def forward(self, image, target):
        return self.model(image, target)


# Custom Model Template
class MyModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        """
        1. 위와 같이 생성자의 parameter 에 num_claases 를 포함해주세요.
        2. 나만의 모델 아키텍쳐를 디자인 해봅니다.
        3. 모델의 output_dimension 은 num_classes 로 설정해주세요.
        """

    def forward(self, x):
        """
        1. 위에서 정의한 모델 아키텍쳐를 forward propagation 을 진행해주세요
        2. 결과로 나온 output 을 return 해주세요
        """
        return x

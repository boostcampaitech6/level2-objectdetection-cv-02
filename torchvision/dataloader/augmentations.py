import albumentations as A
from albumentations.pytorch import ToTensorV2



class CustomAugmentation:
    """커스텀 Augmentation을 담당하는 클래스"""

    def __init__(self, resize, **args):
        self.transform = A.Compose([
            A.Resize(*resize),
            A.Flip(p=0.5),
            ToTensorV2(p=1.0)
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})


    def __call__(self, image, bboxes, labels):
        return self.transform(image=image, bboxes=bboxes, labels=labels)
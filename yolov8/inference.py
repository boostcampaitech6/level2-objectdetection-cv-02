from pycocotools.coco import COCO
from ultralytics import YOLO
import numpy as np
import os

# import albumentations as A
# from albumentations.pytorch import ToTensorV2
import torch

from torch.utils.data import DataLoader, Dataset
import pandas as pd
from tqdm import tqdm


from ultralytics import YOLO

class CustomDataset(Dataset):
    '''
      data_dir: data가 존재하는 폴더 경로
      transforms: data transform (resize, crop, Totensor, etc,,,)
    '''

    def __init__(self, annotation, data_dir):
        super().__init__()
        self.data_dir = data_dir
        # coco annotation 불러오기 (coco API)
        self.coco = COCO(annotation)

    # def __getitem__(self, index: int):
        
    #     image_id = self.coco.getImgIds(imgIds=index)

    #     image_info = self.coco.loadImgs(image_id)[0]
        
    #     image = cv2.imread(os.path.join(self.data_dir, image_info['file_name'].split('/')[1]))
    #     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
    #     # image /= 255.0

    #     # ann_ids = self.coco.getAnnIds(imgIds=image_info['id'])
    #     # anns = self.coco.loadAnns(ann_ids)

    #     # image = torch.tensor(image, dtype=torch.float32).permute(2,0,1)

    #     return image
    
    def __getitem__(self, index: int):
        
        image_id = self.coco.getImgIds(imgIds=index)
        image_info = self.coco.loadImgs(image_id)[0]
        image = os.path.join(self.data_dir, image_info['file_name'].split('/')[1])

        return image
    
    def __len__(self) -> int:
        return len(self.coco.getImgIds())
    

# def inference_fn(test_data_loader, model, device):
#     outputs = []
#     for images in tqdm(test_data_loader):
#         # gpu 계산을 위해 image.to(device)
#         images = list(image.to(device) for image in images)
#         output = model.inference(images)
#         for out in output:
#             outputs.append({'boxes': out['boxes'].tolist(), 'scores': out['scores'].tolist(), 'labels': out['labels'].tolist()})
#     return outputs

def make_submission_file(data_loader, model):
    prediction_strings = []
    file_names = []
    iou_threshold = 0.05
    
    for image_batch in list(tqdm(data_loader)):
        with torch.no_grad():
            predictions = model(image_batch)
        batch_size = len(image_batch)

        for idx in range(batch_size):
            file_name = os.path.join('test', os.path.basename(image_batch[idx]))
            pred_info = predictions[idx].boxes
            classes = pred_info.cls.tolist()
            bboxes = pred_info.xyxy.tolist()
            conf = pred_info.conf.tolist()

            prediction_string = ''
            for pred_class, conf, box in zip(classes, conf, bboxes):
                if conf < iou_threshold:
                    continue
                
                prediction_string += str(int(pred_class))+ ' ' + str(conf) + ' ' + str(box[0]) + ' '  \
                    + str(box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
            prediction_strings.append(prediction_string)
            file_names.append(file_name)
    
    return prediction_strings, file_names
    

def main():
    device = torch.device('cuda')

    annotation = '/data/ephemeral/home/dataset/test.json' # annotation 경로
    data_dir = '/data/ephemeral/home/dataset/test' # dataset 경로
    
    test_dataset = CustomDataset(annotation, data_dir)
    test_data_loader = DataLoader(
        test_dataset,
        batch_size=16,
        shuffle=False,
        num_workers=4
    )
    
    # score_threshold = 0.05
    check_point = './yolo_train/yolov8x12/weights/best.pt' # 체크포인트 경로
    
    # device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    # print(device)

    # torchvision model 불러오기
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    model = YOLO(check_point)
    model.model.eval()
    # model_module = getattr(module_arch, "fasterrcnn_resnet50_fpn")
    # model = model_module(num_classes=11).to(device)
    # model.load_state_dict(torch.load(check_point))
    # model.eval()

    # outputs = inference_fn(test_data_loader, model, device)
    # prediction_strings = []
    # file_names = []
    # coco = COCO(annotation)

    # submission 파일 생성
    # for i, output in enumerate(outputs):
    #     prediction_string = ''
    #     image_info = coco.loadImgs(coco.getImgIds(imgIds=i))[0]
    #     for box, score, label in zip(output['boxes'], output['scores'], output['labels']):
    #         if score > score_threshold: 
    #             # label[1~10] -> label[0~9]
    #             prediction_string += str(label-1) + ' ' + str(score) + ' ' + str(box[0]) + ' ' + str(
    #                 box[1]) + ' ' + str(box[2]) + ' ' + str(box[3]) + ' '
    #     prediction_strings.append(prediction_string)
    #     file_names.append(image_info['file_name'])

    prediction_strings, file_names = make_submission_file(test_data_loader, model)
    submission = pd.DataFrame()
    submission['PredictionString'] = prediction_strings
    submission['image_id'] = file_names
    submission.to_csv('./yolo_submission.csv', index=None)
    print(submission.head())


if __name__ == '__main__':
    main()
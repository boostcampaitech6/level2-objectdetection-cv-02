import numpy as np
import torch
import os
import random
import glob
import re
import json
import wandb
from pathlib import Path
from torchvision.utils import make_grid
from base.base_trainer import BaseTrainer
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from utils import Averager


class Trainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, criterion, optimizer, config, 
                 device=None, train_dataloader=None, valid_dataloader=None, lr_scheduler=None):
        super().__init__(model, criterion, optimizer, config)
        self.device = device
        self.train_dataloader = train_dataloader
        self.valid_dataloader = valid_dataloader
        self.lr_scheduler = lr_scheduler
        self.loss_hist = Averager()
        
        self.best_train_loss = np.inf  # validataion set 만들기 전 임시용
        self.best_val_loss = np.inf

        # logging with wandb
        wandb.init(project="object_detection")
        # 실행 이름 설정
        wandb.run.name = self.config.wandb
        wandb.run.save()
        wandb.config.update(self.config)

        # Wandb run name으로 이후 수정
        self.save_dir = self.increment_path(os.path.join(self.config.model_dir, self.config.name))

        # logging with tensorboard
        self.logger = SummaryWriter(log_dir=self.save_dir)
        with open(os.path.join(self.save_dir, "config.json"), "w", encoding="utf-8") as f:
            json.dump(vars(config), f, ensure_ascii=False, indent=4)


    def increment_path(self, path, exist_ok=False):
        """Automatically increment path, i.e. runs/exp --> runs/exp0, runs/exp1 etc.

        Args:
            path (str or pathlib.Path): f"{model_dir}/{args.name}".
            exist_ok (bool): whether increment path (increment if False).
        """
        path = Path(path)
        if (path.exists() and exist_ok) or (not path.exists()):
            return str(path)
        else:
            dirs = glob.glob(f"{path}*")
            matches = [re.search(rf"%s(\d+)" % path.stem, d) for d in dirs]
            i = [int(m.groups()[0]) for m in matches if m]
            n = max(i) + 1 if i else 2
            return f"{path}{n}"

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group["lr"]

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        """
        self.model.train()
        self.loss_hist.reset()
        
        for idx, train_batch in enumerate(self.train_dataloader):
            images, targets, image_ids = train_batch

            # gpu 계산을 위해 image.to(device)
            images = list(image.float().to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            # calculate loss
            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            loss_value = losses.item()
            loss_classifier = loss_dict['loss_classifier'].item()
            loss_box_reg = loss_dict['loss_box_reg'].item()
            loss_objectness = loss_dict['loss_objectness'].item()
            loss_rpn_box_reg = loss_dict['loss_rpn_box_reg'].item()

            self.loss_hist.send(loss_value)

            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            
            if (idx + 1) % self.config.log_interval == 0:
                train_loss = self.loss_hist.value
                current_lr = self.get_lr(self.optimizer)
                print(
                    f"Epoch[{epoch}/{self.config.epochs}]({idx + 1}/{len(self.train_dataloader)}) || "
                    f"lr {current_lr} || training loss {train_loss:4.4} || classifier {loss_classifier:2.5} || box reg {loss_box_reg:2.5} || "
                    f"objectness {loss_objectness:2.5} || rpn_box_reg {loss_rpn_box_reg:2.5}"
                )

                # wandb: 학습 단계에서 Loss, Accuracy 로그 저장
                wandb.log({
                    "Train loss": train_loss,
                    "Classifier loss": loss_classifier,
                    "box reg": loss_box_reg,
                    "objectness": loss_objectness,
                    "rpn_box_reg": loss_rpn_box_reg
                })
        
        # 지금은 train에 대해 best를 저장
        # 추후 validation set 만들어지면 validation loss에 대해 best 모델을 저장하도록 수정 예정
        if self.loss_hist.value < self.best_train_loss:
            print(
                f"New best model for train loss : {self.loss_hist.value:4.4}! saving the best model.."
            )
            
            torch.save(self.model.module.state_dict(), f"{self.save_dir}/best_loss.pth")
            self.best_val_loss = min(self.best_val_loss, self.loss_hist.value)

            torch.save(self.model.module.state_dict(), f"{self.save_dir}/last.pth")
            print(
                f"[Val] loss: {self.loss_hist.value:4.4} || "
                f"best loss: {self.best_val_loss:4.4}"
            )

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.valid_dataloader is not None:
            self._valid_epoch(epoch)


    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        """
        
        self.model.eval()
        
        with torch.no_grad():
            print("Calculating validation results...")
            val_loss_items = []

            for val_batch in self.valid_dataloader:
                images, targets, image_ids = val_batch

                # gpu 계산을 위해 image.to(device)
                images = list(image.float().to(self.device) for image in images)
                targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

                # calculate loss
                loss_dict = self.model(images, targets)

                losses = sum(loss for loss in loss_dict.values())
                loss_value = losses.item()
                
                val_loss_items.append(loss_value)


            val_loss = np.sum(val_loss_items) / len(self.valid_dataloader)
            
            if val_loss < self.best_val_loss:
                print(
                    f"New best model for val loss : {val_loss:2.4%}! saving the best model.."
                )
                
                torch.save(self.model.module.state_dict(), f"{self.save_dir}/best_loss.pth")
                self.best_val_loss = min(self.best_val_loss, val_loss)

            torch.save(self.model.module.state_dict(), f"{self.save_dir}/last.pth")
            print(
                f"[Val] loss: {val_loss:4.2} || "
                f"best loss: {self.best_val_loss:4.2}"
            )

            # # wandb: 검증 단계에서 Loss, Accuracy 로그 저장
            # wandb.log({
            #     "Valid loss": val_loss,
            #     "Valid acc" : val_acc,
            #     "results": wandb.Image(figure),
            # })

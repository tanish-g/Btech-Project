import os
import time
import subprocess
import numpy as np
import pandas as pd
import ast
import cv2
import PIL.Image
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import TensorDataset, DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR
from warmup_scheduler import GradualWarmupScheduler
import albumentations
import torch.cuda.amp as amp
import segmentation_models_pytorch as smp
from tqdm import tqdm
import random
import argparse

def seed_torch(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

scaler = amp.GradScaler()
device = torch.device('cuda')
image_folder = 'train'
use_amp = True
data_dir = '/ocean/projects/cis220039p/tgupta1/Btech_project/ranzcr_dataset'
df_train_anno = pd.read_csv(os.path.join(data_dir, 'train_annotations.csv'))
image_size = 1024

class RANZCRDataset(Dataset):

    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        image = cv2.imread(os.path.join(data_dir, image_folder, row.StudyInstanceUID + '.jpg'))[:, :, ::-1]

        if self.mode == 'test':
            mask = None
            res = self.transform(image=image)
        else:
            res = self.transform(image=image, mask=mask)

        image = res['image'].astype(np.float32).transpose(2, 0, 1) / 255.

        if self.mode == 'test':
            return torch.tensor(image)
        else:
            mask = res['mask'].astype(np.float32)
            mask = mask.transpose(2, 0, 1).clip(0, 1)
            return torch.tensor(image), torch.tensor(mask)

class SegModel(nn.Module):
    def __init__(self, backbone):
        super(SegModel, self).__init__()
        self.seg = smp.UnetPlusPlus(encoder_name=backbone, encoder_weights='imagenet', classes=2, activation=None)
    def forward(self,x):
        global_features = self.seg.encoder(x)
        seg_features = self.seg.decoder(*global_features)
        seg_features = self.seg.segmentation_head(seg_features)
        return seg_features

class GradualWarmupSchedulerV2(GradualWarmupScheduler):
    def __init__(self, optimizer, multiplier, total_epoch, after_scheduler=None):
        super(GradualWarmupSchedulerV2, self).__init__(optimizer, multiplier, total_epoch, after_scheduler)
    def get_lr(self):
        if self.last_epoch > self.total_epoch:
            if self.after_scheduler:
                if not self.finished:
                    self.after_scheduler.base_lrs = [base_lr * self.multiplier for base_lr in self.base_lrs]
                    self.finished = True
                return self.after_scheduler.get_lr()
            return [base_lr * self.multiplier for base_lr in self.base_lrs]
        if self.multiplier == 1.0:
            return [base_lr * (float(self.last_epoch) / self.total_epoch) for base_lr in self.base_lrs]
        else:
            return [base_lr * ((self.multiplier - 1.) * self.last_epoch / self.total_epoch + 1.) for base_lr in self.base_lrs]

def train_epoch(model, loader, optimizer, criterion):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, mask) in bar:

        optimizer.zero_grad()
        data, mask = data.to(device), mask.to(device)

        with amp.autocast():
            logits = model(data)
            loss = criterion(logits, mask)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        loss_np = loss.item()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-50:]) / min(len(train_loss), 50)
        bar.set_description('loss: %.4f, smth: %.4f' % (loss_np, smooth_loss))
    return np.mean(train_loss)

def iou_score(outputs, targets, eps=1e-7):
    intersection = torch.sum(outputs & targets)
    union = torch.sum(outputs | targets)
    iou = (intersection + eps) / (union + eps)
    return iou.item()

def dice_score(outputs, targets, eps=1e-7):
    intersection = torch.sum(outputs & targets)
    dice = (2. * intersection + eps) / (torch.sum(outputs) + torch.sum(targets) + eps)
    return dice.item()

def valid_epoch(model, loader, criterion, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    iou = []
    dice = []
    with torch.no_grad():
        for (data, mask) in tqdm(loader):
            data, mask = data.to(device), mask.to(device)
            logits = model(data)
            loss = criterion(logits, mask)
            val_loss.append(loss.item())
            LOGITS.append(logits.cpu())
            iou.append(iou_score((logits>0.5).int(),mask.int()))
            dice.append(dice_score((logits>0.5).int(),mask.int()))
    if get_output:
        LOGITS = torch.cat(LOGITS, 0).float().sigmoid()
        return LOGITS,np.array(iou).mean(),np.array(dice).mean()
    else:
        val_loss = np.mean(val_loss)
        return val_loss

if __name__=='__main__':
    #Hyperparams
    # Set the seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--enet_type',type=str,default='3')
    parser.add_argument('--fold',type=int,default=0)

    args = parser.parse_args()

    print(f'Model : efficientnet-b{args.enet_type}')
    print(f'Fold : Fold{args.fold}')

    DEBUG=False
    kernel_type = f'unet++b{args.enet_type}_2cbce_1024T15tip_lr1e4_bs4_augv2_30epo'
    enet_type = f'timm-efficientnet-b{args.enet_type}'
    data_dir = './ranzcr_dataset'
    num_workers = 8
    image_size = 1024
    batch_size = 16
    image_folder = 'train'

    model_dir = './models'
    os.makedirs(model_dir, exist_ok=True)
    # Part 2, For those images without annotations, use 5-fold models to predict and take the mean value
    df_train = pd.read_csv('train_v2.csv')

    # If DEBUG == True, use only 100 samples to run.
    df_train = pd.concat([
        df_train.query('fold == 0').sample(20),
        df_train.query('fold == 1').sample(20),
        df_train.query('fold == 2').sample(20),
        df_train.query('fold == 3').sample(20),
        df_train.query('fold == 4').sample(20),
    ]) if DEBUG else df_train

    df_train_anno = pd.read_csv(os.path.join(data_dir, 'train_annotations.csv'))
    print(df_train.shape)
    transforms_val = albumentations.Compose([albumentations.Resize(image_size, image_size)],is_check_shapes=False)
    df_train_wo_anno = df_train.query(f'w_anno==False').copy().reset_index(drop=True)
    dataset_test = RANZCRDataset(df_train_wo_anno, 'test', transform=transforms_val)
    test_loader = torch.utils.data.DataLoader(dataset_test, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    output_dir = f'mask_{kernel_type}'
    os.makedirs(output_dir, exist_ok=True)
    criterion = nn.BCEWithLogitsLoss()
    # Part 1, generate mask for those images with annotations. To prevent leaks, use only the model corresponding to the fold
    for fold in range(1):
        valid_ = df_train.query(f'w_anno==True and fold=={fold}').copy()
        dataset_valid = RANZCRDataset(valid_, 'valid', transform=transforms_val)
        valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        model = SegModel(enet_type)
        model = model.to(device)
        model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')
        model.load_state_dict(torch.load(model_file), strict=True)
        model.eval()
        
        outputs,score_iou,score_dice = valid_epoch(model, valid_loader, criterion, get_output=True)
        outputs = outputs.numpy()
        print(score_iou)
        print(score_dice)
    
    for i, (_, row) in tqdm(enumerate(valid_.iterrows())):
        png = (outputs[i] * 255).astype(np.uint8).transpose(1,2,0)
        # add a channel to make it able to be saved as .png
        png = np.concatenate([png, np.zeros((png.shape[0], png.shape[1], 1))], -1)
        cv2.imwrite(os.path.join(output_dir, f'{row.StudyInstanceUID}.png'), png)
    
    # models = []

    # for fold in range(1):
    #     model = SegModel(enet_type)
    #     model = model.to(device)
    #     model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')
    #     model.load_state_dict(torch.load(model_file), strict=False)
    #     model.eval()
    #     models.append(model)
    

    # with torch.no_grad():
    #     for batch_id, data in tqdm(enumerate(test_loader), total=len(test_loader)):
    #         data = data.to(device)
    #         outputs = torch.stack([model(data).sigmoid() for model in models], 0).mean(0).cpu().numpy()
    #         for i in range(outputs.shape[0]):
    #             row = df_train_wo_anno.loc[batch_id*batch_size + i]
    #             png = (outputs[i] * 255).astype(np.uint8).transpose(1,2,0)
    #             # add a channel to make it able to be saved as .png
    #             png = np.concatenate([png, np.zeros((png.shape[0], png.shape[1], 1))], -1)
    #             cv2.imwrite(os.path.join(output_dir, f'{row.StudyInstanceUID}.png'), png)

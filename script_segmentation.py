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
            df_this = df_train_anno.query(f'StudyInstanceUID == "{row.StudyInstanceUID}"')
            mask = np.zeros((image.shape[0], image.shape[1], 2)).astype(np.uint8)
            for _, anno in df_this.iterrows():
                anno_this = np.array(ast.literal_eval(anno["data"]))
                mask1 = mask[:, :, 0].copy()
                mask1 = cv2.polylines(mask1, np.int32([anno_this]), isClosed=False, color=1, thickness=15, lineType=16)
                mask[:, :, 0] = mask1
                mask2 = mask[:, :, 1].copy()
                mask2 = cv2.circle(mask2, (anno_this[0][0], anno_this[0][1]), radius=15, color=1, thickness=25)
                mask2 = cv2.circle(mask2, (anno_this[-1][0], anno_this[-1][1]), radius=15, color=1, thickness=25)
                mask[:, :, 1] = mask2

            mask = cv2.resize(mask ,(image_size, image_size))
            mask = (mask > 0.5).astype(np.uint8)
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


def valid_epoch(model, loader, criterion, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    with torch.no_grad():
        for (data, mask) in tqdm(loader):
            data, mask = data.to(device), mask.to(device)
            logits = model(data)
            loss = criterion(logits, mask)
            val_loss.append(loss.item())
            LOGITS.append(logits.cpu())

    if get_output:
        LOGITS = torch.cat(LOGITS, 0).float().sigmoid()
        return LOGITS
    else:
        val_loss = np.mean(val_loss)
        return val_loss

def run(enet_number,fold):

    DEBUG=False
    kernel_type = f'unet++b{enet_number}_2cbce_1024T15tip_lr1e4_bs4_augv2_30epo'
    enet_type = f'timm-efficientnet-b{enet_number}'
    data_dir = '/ocean/projects/cis220039p/tgupta1/Btech_project/ranzcr_dataset'
    num_workers = 4
    image_size = 1024
    batch_size = 8
    init_lr = 1.4e-4
    warmup_epo = 1

    # If DEBUG == True, only run 3 epochs per fold
    cosine_epo = 29 if not DEBUG else 2
    n_epochs = warmup_epo + cosine_epo
    use_amp = True
    image_folder = 'train'

    log_dir = './logs'
    model_dir = './models'
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(model_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'log_unet_arch_{kernel_type}.txt')

    df_train = pd.read_csv('/ocean/projects/cis220039p/tgupta1/Btech_project/train_v2.csv')

    # If DEBUG == True, use only 100 samples to run.
    df_train = pd.concat([
        df_train.query('fold == 0').sample(20),
        df_train.query('fold == 1').sample(20),
        df_train.query('fold == 2').sample(20),
        df_train.query('fold == 3').sample(20),
        df_train.query('fold == 4').sample(20),
    ]) if DEBUG else df_train

    print(df_train.shape)
    content = 'Fold: ' + str(fold)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')
    train_ = df_train.query(f'w_anno==True and fold!={fold}').copy()
    valid_ = df_train.query(f'w_anno==True and fold=={fold}').copy()
    
    transforms_train = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.2, rotate_limit=30, border_mode=0, p=0.75),
    albumentations.CoarseDropout(max_height=int(image_size * 0.3), max_width=int(image_size * 0.3), max_holes=1,min_holes=1, p=0.75)
],is_check_shapes=False)
    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
    ],is_check_shapes=False)

    dataset_train = RANZCRDataset(train_, 'train', transform=transforms_train)
    dataset_valid = RANZCRDataset(valid_, 'valid', transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers,drop_last=True)

    model = SegModel(enet_type)
    model = model.to(device)
    val_loss_min = np.Inf
    model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')
    
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)

    criterion = nn.BCEWithLogitsLoss()

    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch-1)

        train_loss = train_epoch(model, train_loader, optimizer, criterion)
        val_loss = valid_epoch(model, valid_loader, criterion)

        content = time.ctime() + ' ' + f'Fold {fold} Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.5f}, valid loss: {(val_loss):.5f}.'
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if val_loss_min > val_loss:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(val_loss_min, val_loss))
            torch.save(model.state_dict(), model_file)
            val_loss_min = val_loss

if __name__=='__main__':
    #Hyperparams
    # Set the seed
    parser = argparse.ArgumentParser()
    parser.add_argument('--enet_type',type=str,default='3')
    parser.add_argument('--fold',type=int,default=0)

    args = parser.parse_args()

    print(f'Model : efficientnet-b{args.enet_type}')
    print(f'Fold : Fold{args.fold}')

    seed_torch(seed=42)
    run(args.enet_type,args.fold)

import os
import time
import random
import numpy as np
import pandas as pd
import subprocess
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import ast
import seaborn as sns
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
from warmup_scheduler import GradualWarmupScheduler
import albumentations
import timm
from tqdm import tqdm
import torch.cuda.amp as amp
import warnings
import argparse

warnings.simplefilter('ignore')
scaler = amp.GradScaler()
device = torch.device('cuda')


DEBUG=False
kernel_type = 'enetb1_5ch_512_lr3e4_bs32_30epo'
enet_type = 'tf_efficientnet_b1_ns'
data_dir = './ranzcr_dataset'
num_workers = 2
num_classes = 12
# n_ch = 5
n_ch = 3
image_size = 512
batch_size = 32
init_lr = 3e-4
warmup_epo = 1
# If DEBUG == True, only run 3 epochs per fold
cosine_epo = 29 if not DEBUG else 2
n_epochs = warmup_epo + cosine_epo
loss_weights = [1., 9.]
image_folder = 'train'
mask_folder = 'mask_unet++b1_2cbce_1024T15tip_lr1e4_bs4_augv2_30epo/'
df_train_anno = pd.read_csv(os.path.join(data_dir, 'train_annotations.csv'))
log_dir = './logs'
model_dir = './models'
os.makedirs(log_dir, exist_ok=True)
os.makedirs(model_dir, exist_ok=True)
log_file = os.path.join(log_dir, f'log_{kernel_type}.txt')

class enetv2(nn.Module):
    def __init__(self, enet_type, out_dim):
        super(enetv2, self).__init__()
        self.enet = timm.create_model(enet_type, True)
        self.dropout = nn.Dropout(0.5)
        self.enet.conv_stem.weight = nn.Parameter(self.enet.conv_stem.weight.repeat(1,n_ch//3+1,1,1)[:, :n_ch])
        self.myfc = nn.Linear(self.enet.classifier.in_features, out_dim)
        self.enet.classifier = nn.Identity()

    def extract(self, x):
        return self.enet(x)

    def forward(self, x):
        x = self.extract(x)
        h = self.myfc(self.dropout(x))
        return h

class RANZCRDatasetCLS(Dataset):

    def __init__(self, df, mode, transform=None):

        self.df = df.reset_index(drop=True)
        self.mode = mode
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index):
        row = self.df.iloc[index]
        # image = cv2.imread(os.path.join(data_dir, image_folder, row.StudyInstanceUID + '.jpg'))[:, :, ::-1]
        image = np.array(Image.open(os.path.join(data_dir, image_folder, row.StudyInstanceUID + '.jpg')))
        image = np.expand_dims(image,axis=2)
        if row['w_anno']==False:
            mask = cv2.imread(os.path.join(mask_folder, row.StudyInstanceUID + '.png')).astype(np.float32)[:,:,:2]
            res = self.transform(image=image, mask=mask)
            image = res['image'].astype(np.float32).transpose(2, 0, 1) / 255.
            mask = res['mask'].astype(np.float32).transpose(2, 0, 1) / 255.
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
            mask = res['mask'].astype(np.float32).transpose(2, 0, 1).clip(0, 1)

        image = np.concatenate([image, mask], 0)

        if self.mode == 'test':
            return torch.tensor(image)
        else:
            label = row[[
                'ETT - Abnormal',
                'ETT - Borderline',
                'ETT - Normal',
                'no_ETT',
                'NGT - Abnormal',
                'NGT - Borderline',
                'NGT - Incompletely Imaged',
                'NGT - Normal',
                'CVC - Abnormal',
                'CVC - Borderline',
                'CVC - Normal',
                'Swan Ganz Catheter Present'
            ]].values.astype(float)
            return torch.tensor(image).float(), torch.tensor(label).float()

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


bce = nn.BCEWithLogitsLoss()
ce = nn.CrossEntropyLoss()

def criterion(logits, targets, lw=loss_weights):
    loss1 = ce(logits[:, :4], targets[:, :4].argmax(1)) * lw[0]
    loss2 = bce(logits[:, 4:], targets[:, 4:]) * lw[1]
    return (loss1 + loss2) / sum(lw)

def train_epoch(model, loader, optimizer):

    model.train()
    train_loss = []
    bar = tqdm(loader)
    for (data, targets) in bar:

        optimizer.zero_grad()
        data, targets = data.to(device), targets.to(device)

        with amp.autocast():
            logits = model(data)
            loss = criterion(logits, targets)
        scaler.scale(loss).backward() 
        scaler.step(optimizer)
        scaler.update()

        loss_np = loss.item()
        train_loss.append(loss_np)
        smooth_loss = sum(train_loss[-50:]) / min(len(train_loss), 50)
        bar.set_description('loss: %.4f, smth: %.4f' % (loss_np, smooth_loss))

    return np.mean(train_loss)


def valid_epoch(model, loader, get_output=False):

    model.eval()
    val_loss = []
    LOGITS = []
    TARGETS = []
    with torch.no_grad():
        for (data, targets) in tqdm(loader):
            data, targets = data.to(device), targets.to(device)
            logits = model(data)
            loss = criterion(logits, targets)
            val_loss.append(loss.item())
            LOGITS.append(logits.cpu())
            TARGETS.append(targets.cpu())
            
    val_loss = np.mean(val_loss)
    LOGITS = torch.cat(LOGITS)
    LOGITS[:, :4] = LOGITS[:, :4].softmax(1)
    LOGITS[:, 4:] = LOGITS[:, 4:].sigmoid()
    TARGETS = torch.cat(TARGETS).numpy()

    if get_output:
        return LOGITS
    else:
        aucs = []
        for cid in range(num_classes):
            if cid == 3: continue
            try:
                aucs.append( roc_auc_score(TARGETS[:, cid], LOGITS[:, cid]) )
            except:
                aucs.append(0.5)
        return val_loss, aucs

def run(enet_number,fold):
    DEBUG=False
    kernel_type = f'enetb{enet_number}_3ch_512_lr3e4_bs32_30epo'
    enet_type = f'tf_efficientnet_b{enet_number}_ns'
    num_workers = 2
    num_classes = 12
    image_size = 512
    batch_size = 32
    init_lr = 3e-4
    warmup_epo = 1
    # If DEBUG == True, only run 3 epochs per fold
    cosine_epo = 29 if not DEBUG else 2
    n_epochs = warmup_epo + cosine_epo
    df_train = pd.read_csv('train_v2.csv')
    # If DEBUG == True, use only 100 samples to run.
    df_train = pd.concat([
        df_train.query('fold == 0').sample(20),
        df_train.query('fold == 1').sample(20),
        df_train.query('fold == 2').sample(20),
        df_train.query('fold == 3').sample(20),
        df_train.query('fold == 4').sample(20),
    ]) if DEBUG else df_train

    no_ETT = (df_train[['ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal']].values.max(1) == 0).astype(int)
    df_train.insert(4, column='no_ETT', value=no_ETT)

    content = 'Fold: ' + str(fold)
    print(content)
    with open(log_file, 'a') as appender:
        appender.write(content + '\n')
    
    train_ = df_train.query(f'fold!={fold}').copy()
    valid_ = df_train.query(f'fold=={fold}').copy()
    
    transforms_train = albumentations.Compose([
    albumentations.Resize(image_size, image_size),
    albumentations.HorizontalFlip(p=0.5),
    albumentations.OneOf([
        albumentations.OpticalDistortion(distort_limit=1.),
        albumentations.GridDistortion(num_steps=5, distort_limit=1.),
    ], p=0.75),

    albumentations.HueSaturationValue(hue_shift_limit=40, sat_shift_limit=40, val_shift_limit=0, p=0.75),
    albumentations.ShiftScaleRotate(shift_limit=0.2, scale_limit=0.3, rotate_limit=30, border_mode=0, p=0.75),
    albumentations.CoarseDropout(max_height=int(image_size * 0.4), max_width=int(image_size * 0.4), max_holes=1,min_holes=1, p=0.75)
],is_check_shapes=False)
    
    transforms_val = albumentations.Compose([
        albumentations.Resize(image_size, image_size),
    ],is_check_shapes=False)

    dataset_train = RANZCRDatasetCLS(train_, 'train', transform=transforms_train)
    dataset_valid = RANZCRDatasetCLS(valid_, 'valid', transform=transforms_val)
    train_loader = torch.utils.data.DataLoader(dataset_train, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    valid_loader = torch.utils.data.DataLoader(dataset_valid, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    model = enetv2(enet_type, num_classes)
    model = model.to(device)
    aucs_max = 0
    model_file = os.path.join(model_dir, f'{kernel_type}_best_fold{fold}.pth')

    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    scheduler_cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cosine_epo)
    scheduler_warmup = GradualWarmupSchedulerV2(optimizer, multiplier=10, total_epoch=warmup_epo, after_scheduler=scheduler_cosine)
    for epoch in range(1, n_epochs+1):
        print(time.ctime(), 'Epoch:', epoch)
        scheduler_warmup.step(epoch-1)

        train_loss = train_epoch(model, train_loader, optimizer)
        val_loss, aucs = valid_epoch(model, valid_loader)

        content = time.ctime() + ' ' + f'Fold {fold} Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, train loss: {train_loss:.4f}, valid loss: {(val_loss):.4f}, aucs: {np.mean(aucs):.4f}.'
        content += '\n' + ' '.join([f'{x:.4f}' for x in aucs])
        print(content)
        with open(log_file, 'a') as appender:
            appender.write(content + '\n')

        if aucs_max < np.mean(aucs):
            print('aucs increased ({:.6f} --> {:.6f}).  Saving model ...'.format(aucs_max, np.mean(aucs)))
            torch.save(model.state_dict(), model_file)
            aucs_max = np.mean(aucs)

    torch.save(model.state_dict(), os.path.join(model_dir, f'{kernel_type}_model_fold{fold}.pth'))

def seed_torch(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if using multiple GPUs
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--enet_type',type=str,default='3')
    parser.add_argument('--fold',type=int,default=0)

    args = parser.parse_args()

    print(f'Model : efficientnet-b{args.enet_type}')
    print(f'Fold : Fold{args.fold}')

    seed_torch(42)

    run(args.enet_type,args.fold)

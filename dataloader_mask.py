#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 01:23:17 2021

@author: abas
"""


import numpy as np # linear algebra

import torch
from PIL import Image

# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory
import config
import os


from dataimbalance import generate_target,generate_label,generate_box

path=config.path
class MaskDataset(object):
    def __init__(self, transforms=None,transforms2=None,phase='train'):
        self.transforms = transforms
        self.transforms2=transforms2
        self.count=0
        # load all image files, sorting them to
        # ensure that they are aligned
        if phase =='train':
            self.pt='images/'
            self.imgs = list(sorted(os.listdir(path+config.train_path)))
        elif phase=='test':
            self.pt=config.test_path
            self.count=816
            self.imgs = list(sorted(os.listdir(path+config.test_path)))
            
            print(len(self.imgs))
#         self.labels = list(sorted(os.listdir("/kaggle/input/face-mask-detection/annotations/")))
        
    def __getitem__(self, idx):
        # load images ad masks
        file_image = 'maksssksksss'+ str(idx+self.count) + '.png'
        file_label = 'maksssksksss'+ str(idx+self.count) + '.xml'
        img_path = os.path.join(path+self.pt, file_image)
        label_path = os.path.join(path+config.annotation_path, file_label)
        img = Image.open(img_path).convert("RGB")
        #Generate Label
        target = generate_target(idx, label_path)
        
        if self.transforms is not None:
            
            img=self.transforms2(np.array(img))
            augmented = self.transforms(image=img.permute(1,2,0).numpy(), bboxes=target['boxes'], labels=target['labels'])
            img=self.transforms2(augmented['image'])
            labels=augmented['labels']
            labels[labels==0]=3
            return img, {'boxes':torch.floor(torch.tensor(augmented['bboxes'])).type(torch.int32),
                         'labels':torch.tensor(labels),'image_id':target['image_id']}
        
        else:
            img=self.transforms2(np.array(img))
            return img, target
        
        #return img,target
        

    def __len__(self):
        return len(self.imgs)-1

# %% [code]


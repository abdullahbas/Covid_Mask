#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:52:28 2021

@author: abas
"""
import numpy as np # linear algebra
from bs4 import BeautifulSoup
import torch
import matplotlib.pyplot as plt

import os

path='archive/'
def generate_box(obj):
    
    xmin = int(obj.find('xmin').text)
    ymin = int(obj.find('ymin').text)
    xmax = int(obj.find('xmax').text)
    ymax = int(obj.find('ymax').text)
    
    return [xmin, ymin, xmax, ymax]

def generate_label(obj):
    if obj.find('name').text == "with_mask":
        return 1
    elif obj.find('name').text == "mask_weared_incorrect":
        return 2
    return 0

def generate_target(image_id, file): 
    with open(file) as f:
        data = f.read()
        soup = BeautifulSoup(data, 'xml')
        objects = soup.find_all('object')

        num_objs = len(objects)

        # Bounding boxes for objects
        # In coco format, bbox = [xmin, ymin, width, height]
        # In pytorch, the input should be [xmin, ymin, xmax, ymax]
        boxes = []
        labels = []
        area=[]
        for i in objects:
            boxes.append(generate_box(i))
            labels.append(generate_label(i))
            area.append(61)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # Labels (In my case, I only one class: target class or background)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(area, dtype=torch.int64)
        # Tensorise img_id
        img_id = torch.tensor([image_id])
        # Annotation is in dictionary format
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = img_id
        target["area"]=area
        
        return target
    
if __name__ == '__main__':
    imgs = list(sorted(os.listdir(path+"images/")))
    labels = list(sorted(os.listdir(path+"annotations/")))
    
    # %% [code]
    
    wmask=0 #without mask class counter
    mask=0 # mask class counter
    maskI=0 # Incorrect mask class counter
    for idx,samples in enumerate(labels):
        
        target=generate_target(idx,path+'annotations/'+samples)
        labs=target['labels']
        labs=np.append(labs,[0,1,2])
        counts=np.unique(labs,return_counts=True)[1]
        wmask+=counts[0]-1
        mask+=counts[1]-1
        maskI+=counts[2]-1
        
    fig=plt.figure()
    langs = ['Without Mask','Masked','Improper Masked']
    students = [wmask,mask,maskI]
    plt.bar(langs,students)
    plt.show()
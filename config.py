#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  1 12:55:07 2021

@author: abas
"""
import torch 

path='archive/'

num_classes = 4

num_epochs = 25

batch_size = 1

learning_rate = 0.001

device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

train_path='images/'

test_path='test_images/'

annotation_path="annotations/"
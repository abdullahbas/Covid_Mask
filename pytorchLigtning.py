import torch
#import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
#import torch.nn.functional as F
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import numpy as np
import albumentations as A
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
#from engine import train_one_epoch, evaluate
import os
from PIL import Image
#from pl_bolts.callbacks import TensorboardGenerativeModelImageSampler
import matplotlib.patches as patches

import config
from dataloader_mask import MaskDataset
# Device configuration

# Hyper-parameters
path=config.path
num_classes = config.num_classes
num_epochs = config.num_epochs
batch_size = config.batch_size
learning_rate =config.learning_rate


device = config.device

data_transform = transforms.Compose([
        transforms.ToTensor(), 
    ])

bbox_params = A.BboxParams(
  format='pascal_voc', 
  min_area=1, 
  min_visibility=0.5, 
  label_fields=['labels']
)

aug = A.Compose({
        #A.Resize(500, 500,p=0.2),
        #A.RGBShift(r_shift_limit=40,g_shift_limit=40,b_shift_limit=40,p=0.04),
        #A.RandomBrightness(p=0.01),
        A.RandomContrast(p=0.01),
        #A.CLAHE(p=0.2),
        A.ToGray(p=0.4),
        A.Blur(blur_limit=3,p=0.1),
        #A.RandomBrightness(p=0.1),
        #A.CenterCrop(100, 100,p=0.1),
        #A.RandomCrop(80, 80,p=0.2),
        #A.HorizontalFlip(p=0.1),
        #A.Rotate(limit=(-20, 20),p=0.1),
        #A.VerticalFlip(p=0.1),
        A.ChannelShuffle(p=0.01),
        
        },bbox_params=bbox_params)



def accuracyMetric(preds,annotations):
    non_accurate=0
    accurate=0
    def csm(A,B,corr):
        if corr:
            B=B-B.mean(axis=1)[:,np.newaxis]
            A=A-A.mean(axis=1)[:,np.newaxis]
        num=np.dot(A,B.T)
        p1=np.sqrt(np.sum(A**2,axis=1))[:,np.newaxis]
        p2=np.sqrt(np.sum(B**2,axis=1))[np.newaxis,:]
        return 1-(num/(p1*p2))
    inds=torch.where((preds['scores'])>0.91)
    distMatrix=csm(np.array(preds['boxes'][inds].cpu()),np.array(annotations['boxes'].cpu()),True)
    
    for i in range (distMatrix.shape[0]):
        cla=np.argmin(distMatrix[i,:])
        
        if preds['labels'][i]%3==annotations['labels'][cla]:
            accurate+=1
        else:
            non_accurate+=1
    allSamp=np.max(((accurate+non_accurate),len(annotations['labels'])))
    return (accurate/allSamp)




def plot_image(img_tensor, annotation,phase='train'):
    
    fig,ax = plt.subplots(1)
    img = img_tensor.cpu().data
    
                    
    # Display the image
    ax.imshow(img.permute(1, 2, 0))
    for idx,box in enumerate(annotation["boxes"]):
      
       
      if phase=='test':
        test=1
      else:
        test=annotation['scores'][idx] 
      if test>0.95 :
        xmin, ymin, xmax, ymax = box
        color=['r','g','b','r']
        classes=['no mask','Masked','Improper masking','No-mask']
        # Create a Rectangle patch
        rect = patches.Rectangle((xmin,ymin),(xmax-xmin),(ymax-ymin),linewidth=3,edgecolor=color[annotation['labels'][idx]],facecolor='none')
        ax.text(xmin, ymin, classes[annotation['labels'][idx]],color='black',bbox=dict(facecolor=color[annotation['labels'][idx]], alpha=0.8))
        # Add the patch to the Axes
        ax.add_patch(rect)
      
    plt.show()


def collate_fn(batch):
    return tuple(zip(*batch))

dataset = MaskDataset(aug,data_transform)
data_loader = torch.utils.data.DataLoader(
 dataset, batch_size=batch_size, collate_fn=collate_fn)



def get_model_instance_segmentation(num_classes):
    # load an instance segmentation model pre-trained pre-trained on COCO
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    # get number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    # replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model

# %% [code]




# Fully connected neural network with one hidden layer
class LitNeuralNet(pl.LightningModule):
    def __init__(self, num_classes):
        super(LitNeuralNet, self).__init__()
        self.model = get_model_instance_segmentation(num_classes)
    def forward(self, x,annotations=None,phase='train'):
        
        if phase=='train':
            
            out=self.model(x,annotations)
        else:
            self.model.eval()
            out=model(x)
        # no activation and no softmax at the end
        return out
    def training_step(self, batch, batch_idx):
        imgs, annotations = batch
        imgs = list(img for img in imgs)
        annotations = [{k: v for k, v in t.items()} for t in annotations]

        # Forward pass
        loss_dict = self(imgs,annotations)
        losses = sum(loss for loss in loss_dict.values())        
        occurrences = np.count_nonzero(annotations[0]['labels'].cpu() == 2)
        occurrences2=np.count_nonzero(annotations[0]['labels'].cpu() == 1)
        occurrences=occurrences/(occurrences2+1 )           

        if occurrences>=1:
    
            occurrences=np.clip(occurrences,1,4)
            loss_dict['loss_classifier']=occurrences*4*loss_dict['loss_classifier']
            print(f'Weighted {occurrences}')
            
        elif losses<0.2:
            for k,v in zip(loss_dict,loss_dict.values()):
                loss_dict[k]=v*0
        loss = sum(loss for loss in loss_dict.values())
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        tensorboard_logs = {'train_loss': loss , 'classifier_loss': loss_dict['loss_classifier'],
                            'box_reg_loss':loss_dict['loss_box_reg']}
        # use key 'log'
        
        
            
            
        
        
        return {"loss": loss, 'log': tensorboard_logs}
    def test_step(self,batch,batch_idx):
        imgs, annotations = batch
        #imgs = list(img for img in imgs)
        annotations = [{k: v for k, v in annotations.items()}]
        preds = self(list([imgs]),phase='test')
        #print(preds['labels'])
        
        plot_image(imgs, preds[0])
        plot_image(imgs, annotations[0],phase='test')
        self.log('accuracy',(accuracyMetric(preds[0], annotations[0])))
        print((accuracyMetric(preds[0], annotations[0])))
        
    
    # define what happens for testing here
    def log_hyperparams(self, params):
        epoch=num_epochs
      
        
        return {"epoch": epoch, 'lr': learning_rate,'batch':batch_size}
    def train_dataloader(self):
        # MNIST dataset
        def collate_fn(batch):
            return tuple(zip(*batch))

        dataset = MaskDataset(aug,data_transform)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn)

        return data_loader
    def training_epoch_end(self, outputs):
        avg_val_loss = torch.stack([x['loss'] for x in outputs]).mean()
        print(avg_val_loss)
        return {
            'avg_train_loss': avg_val_loss
        }
    
    
    def val_dataloader(self):
        def collate_fn(batch):
            return tuple(zip(*batch))

        dataset = MaskDataset(transforms2=data_transform,phase='test')
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn)

        return data_loader
    
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=learning_rate,
                                momentum=0.9, weight_decay=0.0005)

if __name__ == '__main__':
    model = LitNeuralNet(num_classes)
    # gpus=8

    # fast_dev_run=True -> runs single batch through training and validation
    # train_percent_check=0.1 -> train only on 10% of data callbacks=[TensorboardGenerativeModelImageSampler()]
    trainer = Trainer(max_epochs=num_epochs,fast_dev_run=False,gpus='0',
                      auto_lr_find=True,deterministic=True,
                      log_gpu_memory=True,precision=32,
                      progress_bar_refresh_rate=20,
                      resume_from_checkpoint='md.ckpt')
    #trainer.fit(model)
 
    checkpoint = torch.load('sonmodel.ckpt')
    model.load_state_dict(checkpoint['state_dict'])
    dataset = MaskDataset(transforms2=data_transform,phase='test')
    data_loader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, collate_fn=collate_fn)

    trainer.test(model,test_dataloaders=dataset)    

    


    

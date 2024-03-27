import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision.transforms import Resize, PILToTensor, Compose, InterpolationMode
from PIL import Image
import os
import matplotlib.pyplot as plt
import numpy as np
class UNetDataClass(Dataset):
    def __init__(self, images_path, masks_path,normal_transform, transform = None, is_train = True ):
        super(UNetDataClass, self).__init__()
        
        images_list = os.listdir(images_path)
        masks_list = os.listdir(masks_path)
        
        images_list = [images_path + image_name for image_name in images_list]
        masks_list = [masks_path + mask_name for mask_name in masks_list]
        
        self.transform = transform
        self.normal_transform = normal_transform

        if is_train:
            self.images_list = images_list[:int(0.85*len(images_list))]
            self.masks_list = masks_list[:int(0.85*len(masks_list))]
        else:
            self.images_list = images_list[int(0.85*len(images_list)):]
            self.masks_list = masks_list[int(0.85*len(masks_list)):]
    
        
    def __getitem__(self, index):
        img_path = self.images_list[index]
        mask_path = self.masks_list[index]
        # Open image and mask
        data = Image.open(img_path)
        label = Image.open(mask_path)
        
        data = np.array(data)
        label = np.array(label)
        # Normalize
        
        if self.transform:
            augmented = self.transform(image = data, mask = label)
        else:
            augmented = self.normal_transform(image = data, mask = label)
        data= augmented["image"]
        label = augmented["mask"]
        data = data / 255
        label = label / 255
        
        
        
        label = torch.where(label>0.1, 1.0, 0.0)
        label[:, :, 2] = 0.0001
        label = torch.argmax(label, 2).type(torch.int64)
        return data, label
    
    def __len__(self):
        return len(self.images_list)
    
    
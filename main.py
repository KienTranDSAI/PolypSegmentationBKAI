from torchsummary import summary
from torchgeometry.losses import one_hot
import os
import pandas as pd
import numpy as np
from PIL import Image
import cv2
import time
import imageio
import matplotlib.pyplot as plt
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision.transforms as transforms
from torchvision.transforms import Resize, PILToTensor, ToPILImage, Compose, InterpolationMode
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
from collections import OrderedDict
import wandb

from data.dataloader import UNetDataClass
from data.test_dataloader import UNetTestDataClass
from model.module import *
from model.unet import UNet
from model.resunet import ResUnet
from model.loss import *

from utilities import utils, test, train, arg_parser

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

args = arg_parser.parser.parse_args()

# Set hyperparameters
model_name = args.model
loss_fc_name = args.loss_fc
optimizer_name = args.optimizer

num_classes = 3
epochs = args.num_epochs
learning_rate = args.lr
batch_size = args.batch_size
display_step = args.display_step
train_size = args.train_size
valid_size = args.valid_size

checkpoint_path = args.checkpoint_path
pretrained_path = args.pretrained_path
inference_path = args.infer_path
images_path = args.images_path
masks_path =  args.masks_path
test_images_path = args.test_path

loss_epoch_array = []
train_accuracy = []
test_accuracy = []
valid_accuracy = []
train_loss_array = []
test_loss_array = []
last_loss = 9999999999999

# Dataloader
print(f"[PROGRESS] STEP 1: Loading data...")
# transform = Compose([Resize((256, 256), interpolation=InterpolationMode.BILINEAR),
#                      PILToTensor()])
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomVerticalFlip(p=0.5),
    transforms.GaussianBlur(3),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

unet_dataset = UNetDataClass(images_path, masks_path, transform)
print(f"[INFO] Size of dataset: {len(unet_dataset)}")
train_set, valid_set = random_split(unet_dataset, 
                                [int(train_size * len(unet_dataset)) , 
                                    int(valid_size * len(unet_dataset))])
train_dataloader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
valid_dataloader = DataLoader(valid_set, batch_size=batch_size, shuffle=True)

print(f"[INFO] Train dataloader: {len(train_dataloader)} batches")
print(f"[INFO] Valid dataloader: {len(valid_dataloader)} batches")
print("="*25 + "END STEP 1" + "="*25)

# Model
print(f"[PROGRESS] STEP 2: Initializing model...")
if (model_name.lower() == "unet"):
    model = UNet()
elif (model_name.lower() == "resunet"):
    model = ResUnet(channel=3, filters=[64, 128, 256, 512])


if (loss_fc_name.lower() == "crossentropyloss"):
    loss_function = CrossEntropyLoss()
else:
    weights = torch.Tensor([[0.4, 0.55, 0.05]]).cuda()
    if (loss_fc_name.lower() == "cediceloss"):
        loss_function = CEDiceLoss(weights)
    elif (loss_fc_name.lower() == "bcediceloss"):
        loss_function = BCEDiceLoss(weights)
print(f"[INFO] Loss function used: {loss_fc_name.lower()}")

model.apply(utils.weights_init)
model = nn.DataParallel(model)
model.to(device)

# Optimizer
if (optimizer_name.lower() == "adam"):
    optimizer = optim.Adam(params=model.parameters(), lr=learning_rate)
learing_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.6)

print("="*25 + "END STEP 2" + "="*25)

# Training
print(f"[PROGRESS] STEP 3: Training...")
wandb.login(key = "09d9c01b499874346601e6ec425a8c58a6f82000")
wandb.init(project="PolypSegment")

for epoch in range(epochs):
    train_loss_epoch = 0
    test_loss_epoch = 0
    (train_loss_epoch, test_loss_epoch) = train.train(model= model, device= device, loss_function= loss_function, optimizer= optimizer, \
                                                    train_dataloader= train_dataloader, valid_dataloader= valid_dataloader, \
                                                    epoch= epoch, display_step= display_step, learing_rate_scheduler= learing_rate_scheduler)
    
    if test_loss_epoch < last_loss:
        utils.save_model(model, optimizer, checkpoint_path)
        last_loss = test_loss_epoch
        
    learing_rate_scheduler.step()
    train_loss_array.append(train_loss_epoch)
    test_loss_array.append(test_loss_epoch)
    wandb.log({"Train loss": train_loss_epoch, "Valid loss": test_loss_epoch})
    # print(f"[INFO] Train loss: {train_loss_epoch}, Valid loss: {test_loss_epoch}")
    train_accuracy.append(test.test(model=model, device=device, dataloader=train_dataloader))
    valid_accuracy.append(test.test(model=model, device=device, dataloader=valid_dataloader))
    print("Epoch {}: loss: {:.4f}, train accuracy: {:.4f}, valid accuracy:{:.4f}".format(epoch + 1, 
                                        train_loss_array[-1], train_accuracy[-1], valid_accuracy[-1]))
    torch.cuda.empty_cache()
print("="*25 + "END STEP 2" + "="*25)

# Results visualization
print(f"[PROGRESS] STEP 3: Plotting diagrams...")
utils.learning_curve_plotting(epochs=epochs, train_loss_array=train_loss_array)
utils.result_visualization(model=model, device=device, train_dataloader=train_dataloader)
print("="*25 + "END STEP 3" + "="*25)

# Testing
print(f"[PROGRESS] STEP 4: Testing...")
unet_test_dataset = UNetTestDataClass(images_path=test_images_path, transform=test_transform)
test_dataloader = DataLoader(unet_test_dataset, batch_size=batch_size, shuffle=True)

# for i, (data, path, h, w) in enumerate(test_dataloader):
#     img = data
#     break

# utils.prediction_visualization(model=model, device=device, img=img)
utils.save_prediction_image(model=model, device=device, test_dataloader=test_dataloader, infer_path=inference_path)
utils.prediction_to_csv(infer_path=inference_path)
print("="*25 + "END STEP 4" + "="*25)
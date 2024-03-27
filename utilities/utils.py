import gc
import os
import cv2
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision.transforms import Resize, ToPILImage, InterpolationMode
from pynvml import nvmlInit, nvmlDeviceGetHandleByIndex, nvmlDeviceGetMemoryInfo
import os
import collections
import datetime

def clear_gpu_memory():
    # torch.cuda.memory_summary(device=None, abbreviated=False)
    torch.cuda.empty_cache()
    gc.collect()
    del variables

def wait_until_enough_gpu_memory(min_memory_available= 2*1024*1024*1024, max_retries=10, sleep_time=5):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(torch.cuda.current_device())

    for _ in range(max_retries):
        info = nvmlDeviceGetMemoryInfo(handle)
        if info.free >= min_memory_available:
            break
        print(f"Waiting for {min_memory_available} bytes of free GPU memory. Retrying in {sleep_time} seconds...")
        clear_gpu_memory()
        time.sleep(sleep_time)
    else:
        raise RuntimeError(f"Failed to acquire {min_memory_available} bytes of free GPU memory after {max_retries} retries.")

def weights_init(model):
    if isinstance(model, nn.Linear):
        # Xavier Distribution
        torch.nn.init.xavier_uniform_(model.weight)

def save_model(model, optimizer, path):
    checkpoint_dir = os.path.dirname(path)
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def load_model(model, optimizer, path):
    checkpoint = torch.load(path)
    try:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint['optimizer'])
    except RuntimeError: 
        # Remove ".module" from the keys of state_dict
        new_state_dict = collections.OrderedDict()
        for k, v in checkpoint["model"].items():
            name = k[7:]
            new_state_dict[name] = v
        model.load_state_dict(new_state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
    return model, optimizer

def learning_curve_plotting(epochs, train_loss_array):
    plt.rcParams['figure.dpi'] = 90
    plt.rcParams['figure.figsize'] = (6, 4)
    epochs_array = range(epochs)
    # Plot Training and Test loss
    plt.plot(epochs_array, train_loss_array, 'g', label='Training loss')
    # plt.plot(epochs_array, test_loss_array, 'b', label='Test loss')
    plt.title('Training and Test loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def result_visualization(model, device, train_dataloader):
    for i, (data, label) in enumerate(train_dataloader):
        img = data.to(device)
        mask = label.to(device)
        break
    fig, arr = plt.subplots(4, 3, figsize=(16, 12))
    arr[0][0].set_title('Image')
    arr[0][1].set_title('Segmentation')
    arr[0][2].set_title('Predict')

    model.eval()
    with torch.no_grad():
        predict = model(img)

    for i in range(4):
        arr[i][0].imshow((img[i].cpu()).permute(1, 2, 0));
        
        arr[i][1].imshow(F.one_hot(mask[i].cpu()).float())
        
        arr[i][2].imshow(F.one_hot(torch.argmax(predict[i], 0).cpu()).float())

def prediction_visualization(model, device, img):
    fig, arr = plt.subplots(5, 2, figsize=(16, 12))
    arr[0][0].set_title('Image');
    arr[0][1].set_title('Predict');

    model.eval()
    with torch.no_grad():
        predict = model(img.to(device))

    for i in range(5):
        arr[i][0].imshow((img[i].cpu()).permute(1, 2, 0));
        arr[i][1].imshow(F.one_hot(torch.argmax(predict[i], 0).cpu()).float())

def save_prediction_image(model, device, test_dataloader, infer_path):
    model.to(device)
    model.eval()
    if not os.path.isdir(infer_path):
        os.mkdir(infer_path)
    for _, (img, path, H, W) in enumerate(test_dataloader):
        a = path
        b = img.to(device)
        h = H
        w = W
        
        with torch.no_grad():
            predicted_mask = model(b)
        for i in range(len(a)):
            image_id = a[i].split('/')[-1].split('.')[0]
            filename = image_id + ".png"
            mask2img = Resize((h[i].item(), w[i].item()), interpolation=InterpolationMode.NEAREST)(ToPILImage()(F.one_hot(torch.argmax(predicted_mask[i], 0)).permute(2, 0, 1).float()))
            mask2img.save(os.path.join(infer_path, filename))



def rle_to_string(runs):
    return ' '.join(str(x) for x in runs)

def rle_encode_one_mask(mask):
    pixels = mask.flatten()
    pixels[pixels > 0] = 255
    use_padding = False
    if pixels[0] or pixels[-1]:
        use_padding = True
        pixel_padded = np.zeros([len(pixels) + 2], dtype=pixels.dtype)
        pixel_padded[1:-1] = pixels
        pixels = pixel_padded
    
    rle = np.where(pixels[1:] != pixels[:-1])[0] + 2
    if use_padding:
        rle = rle - 1
    rle[1::2] = rle[1::2] - rle[:-1:2]
    return rle_to_string(rle)

def mask2string(dir):
    strings = []
    ids = []
    ws, hs = [[] for i in range(2)]
    for image_id in os.listdir(dir):
        id = image_id.split('.')[0]
        path = os.path.join(dir, image_id)
        print(path)
        img = cv2.imread(path)[:,:,::-1]
        h, w = img.shape[0], img.shape[1]
        for channel in range(2):
            ws.append(w)
            hs.append(h)
            ids.append(f'{id}_{channel}')
            string = rle_encode_one_mask(img[:,:,channel])
            strings.append(string)
    r = {
        'ids': ids,
        'strings': strings,
    }
    return r

def prediction_to_csv(infer_path):
    res = mask2string(infer_path)
    df = pd.DataFrame(columns=['Id', 'Expected'])
    df['Id'] = res['ids']
    df['Expected'] = res['strings']
    
    # Generate the file name
    now = datetime.datetime.now()
    file_name = f"output_{now.year}y_{now.month}m_{now.day}d_{now.hour}h_{now.minute}m_{now.second}s.csv"
    
    df.to_csv(file_name, index=False)
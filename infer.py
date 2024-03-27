from utilities import utils
from utilities import arg_parser
from data.test_dataloader import UNetTestDataClass
from torch.utils.data import DataLoader
import torch
from torchvision.transforms import transforms
import numpy as np
from model.unet import UNet
from model.resunet import ResUnet

import torch.optim as optim

def infer_result(pretrained_model_path, model_info, optimizer_info, device, infer_path, transform):
    # Load model
    model, optimizer = utils.load_model(model=model_info, optimizer=optimizer_info, path=pretrained_model_path)
    # Load data
    unet_test_dataset = UNetTestDataClass(infer_path, transform=transform)
    test_dataloader = DataLoader(unet_test_dataset, batch_size=1, shuffle=False)

    utils.save_prediction_image(model=model, device=device, test_dataloader=test_dataloader, infer_path=infer_path)
    utils.prediction_to_csv(infer_path=infer_path)

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    args = arg_parser.parser.parse_args()
    pretrained_model_path = args.checkpoint_path
    learning_rate = args.lr
    inference_path = args.infer_path
    model_name = args.model
    optimizer_name = args.optimizer

    test_transform = transforms.Compose([
        transforms.Resize((256,256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    if (model_name.lower() == "unet"):
        model_info = UNet()
    elif (model_name.lower() == "resunet"):
        model_info = ResUnet(channel=3, filters=[64, 128, 256, 512])
 
    
    if (optimizer_name.lower() == "adam"):
        optimizer_info = optim.Adam(params=model_info.parameters(), lr=learning_rate)

    infer_result(pretrained_model_path=pretrained_model_path, model_info=model_info, optimizer_info=optimizer_info, device= device, infer_path=inference_path, transform=test_transform)
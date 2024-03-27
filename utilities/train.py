import time
import torch
from utilities import utils
from torch.optim import lr_scheduler

# Train function for each epoch
def train(model, device, loss_function, optimizer, train_dataloader, valid_dataloader, epoch, display_step, learing_rate_scheduler=None):
    if learing_rate_scheduler is None:
        learing_rate_scheduler = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.6)
    
    print(f"Start epoch #{epoch+1}, learning rate for this epoch: {learing_rate_scheduler.get_last_lr()}")
    start_time = time.time()
    train_loss_epoch = 0
    test_loss_epoch = 0
    last_loss = 999999999
    model.train()
    for i, (data,targets) in enumerate(train_dataloader):
        
        # Load data into GPU
        data, targets = data.to(device), targets.to(device)

        optimizer.zero_grad()
        try:
            outputs = model(data)
        except RuntimeError as exception:
            if "out of memory" in str(exception):
                print(f"WARNING: out of memory, retrying with 2Gb of free GPU memory...")
                utils.wait_until_enough_gpu_memory()
                outputs = model(data)
            else:
                raise exception

        # Backpropagation, compute gradients
        # try:
        #     loss = loss_function(outputs, targets.float())
        # except:
        #     loss = loss_function(outputs, targets.long())
        loss = loss_function(outputs, targets.long())

        loss.backward()
        optimizer.step()
        
        # Save loss
        train_loss_epoch += loss.item()
        if (i+1) % display_step == 0:
        # accuracy = float(test(test_loader))
            print('Train Epoch: {} [{}/{} ({}%)]\tLoss: {:.4f}'.format(
                epoch + 1, (i+1) * len(data), len(train_dataloader.dataset), 100 * (i+1) * len(data) / len(train_dataloader.dataset), 
                loss.item()))
                  
    print(f"Done epoch #{epoch+1}, time for this epoch: {time.time()-start_time}s")
    train_loss_epoch/= (i + 1)
    
    # Evaluate the validation set
    model.eval()
    with torch.no_grad():
        for data, target in valid_dataloader:
            data, target = data.to(device), target.to(device)
            test_output = model(data)
            test_loss = loss_function(test_output.float(), target)
            test_loss_epoch += test_loss.item()
            
    test_loss_epoch/= (i+1)
    
    return train_loss_epoch , test_loss_epoch
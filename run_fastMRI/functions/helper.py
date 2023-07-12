import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from functions.training.losses import SSIMLoss
from functions.math import complex_abs

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')


#################################################################################################
###         training
#################################################################################################

def adapt(model, dataloader, optimizer): 
    model.train()
    train_loss = 0.0
    mse_fct = torch.nn.MSELoss(reduction='sum')

    for iter, batch in enumerate(dataloader):
        input_image, target_image, mean, std, fname, slice_num = batch
        train_inputs = input_image.to(device)
        train_targets = target_image.to(device)
        std = std.to(device)
        mean = mean.to(device)

        train_outputs = model(train_inputs)
        train_outputs = train_outputs * std + mean

        loss = mse_fct(train_outputs, train_targets) / torch.sum(torch.abs(train_targets)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss


#################################################################################################
###         evaluation
#################################################################################################

def evaluate2c_imagepair(model, input_image, target_image, mean, std, ground_truth_image):
    model.eval()
    l1_loss = torch.nn.L1Loss(reduction='sum')
    ssim_fct = SSIMLoss()
    psner_mse_fct = torch.nn.MSELoss(reduction='mean')
    mse_fct = torch.nn.MSELoss(reduction='sum')

    input_image = input_image.to(device)
    target_image = target_image.to(device)
    std = std.to(device)
    mean = mean.to(device)
    ground_truth_image = ground_truth_image.squeeze(0).to(device)
    
    # time start
    start_time = time.time()
    output_image = model(input_image)
    output_image = output_image * std + mean

    # Move complex dim to end, apply complex abs, insert channel dimension
    output_image_1c = complex_abs(torch.moveaxis(output_image , 1, -1 ))

    # time end
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Inference time: ', elapsed_time)
    # for NMAE/NMSE: complex image [2, 640, 320]
    # NMAE
    nmae_loss = (l1_loss(output_image, target_image) / torch.sum(torch.abs(target_image))).item()
    # NMSE 
    nmse_loss = (mse_fct(output_image, target_image) / torch.sum(torch.abs(target_image)**2)).item()
    # for PSNR/SSIM: real image [1, 640, 320]
    # PSNR
    psnr_loss = (20*torch.log10(torch.tensor(ground_truth_image.max().unsqueeze(0).item()))-10*torch.log10(psner_mse_fct(output_image_1c, ground_truth_image))).item()
    # SSIM = 1 - loss
    ssim_loss = 1 - ssim_fct(output_image_1c, ground_truth_image, data_range = ground_truth_image.max().unsqueeze(0)).item()

    return nmae_loss, nmse_loss, psnr_loss, ssim_loss, output_image, output_image_1c



def evaluate_loss_dataloader(model, dataloader):
    model.eval()
    total_l1_loss = 0.0
    total_ssim_loss = 0.0
    total_psnr_loss = 0.0
    total_nmse_loss = 0.0
    l1_loss = torch.nn.L1Loss(reduction='sum')
    ssim_fct = SSIMLoss()
    psner_mse_fct = torch.nn.MSELoss(reduction='mean')
    mse_fct = torch.nn.MSELoss(reduction='sum')

    for iter, batch in enumerate(dataloader): 
        input_image, target_image, mean, std, fname, slice_num = batch
        val_inputs = input_image.to(device)
        val_targets = target_image.to(device)
        std = std.to(device)
        mean = mean.to(device)

        val_outputs = model(val_inputs)
        val_outputs = val_outputs * std + mean

        # NMAE
        l1 = l1_loss(val_outputs, val_targets) / torch.sum(torch.abs(val_targets))
        total_l1_loss += l1.item()
        # NMSE 
        nmse_loss = mse_fct(val_outputs, val_targets) / torch.sum(torch.abs(val_targets)**2)
        total_nmse_loss += nmse_loss.item()
        # PSNR
        psnr_loss = 20*torch.log10(torch.tensor(val_targets.max().unsqueeze(0).item()))-10*torch.log10(psner_mse_fct(val_outputs,val_targets))
        total_psnr_loss += psnr_loss.item()
        # SSIM = 1 - loss
        ssim_loss = ssim_fct(val_outputs, val_targets, data_range = val_targets.max().unsqueeze(0))
        total_ssim_loss += (1-ssim_loss.item())

    validation_loss_l1 = total_l1_loss / len(dataloader) 
    validation_loss_NMSE = total_nmse_loss / len(dataloader)
    validation_loss_PSNR = total_psnr_loss / len(dataloader)
    validation_loss_SSIM = total_ssim_loss / len(dataloader)

    return validation_loss_l1, validation_loss_NMSE, validation_loss_PSNR, validation_loss_SSIM



#################################################################################################
###         early stopping
#################################################################################################

# for one sampled specific few-shots
def cal_early_stopping(model, checkpoint_path, optimizer, dataset, training_epoch, split_ratio):
    # init the model
    model.load_state_dict(torch.load(checkpoint_path))
    model.train()
    # Split the val data for early stop
    train_dataset, val_dataset = torch.utils.data.random_split(list(dataset), split_ratio)

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset, batch_size = 1,  
                        shuffle = False, generator = torch.Generator().manual_seed(1))
    val_loader = torch.utils.data.DataLoader(dataset = val_dataset, batch_size = 1,
                        shuffle = False, generator = torch.Generator().manual_seed(1))
    val_loss_history = []
    for i in tqdm(range(training_epoch)):    
        #print('Iteration:', iteration+1)
        # train on 5-shots
        mse_fct = torch.nn.MSELoss(reduction='sum')

        for iter, batch in enumerate(train_loader):
            input_image, target_image, mean, std, fname, slice_num = batch
            train_inputs = input_image.to(device)
            train_targets = target_image.to(device)
            std = std.to(device)
            mean = mean.to(device)

            train_outputs = model(train_inputs)
            train_outputs = train_outputs * std + mean

            loss = mse_fct(train_outputs, train_targets) / torch.sum(torch.abs(train_targets)**2)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        # evaluate on another 5-shots
        validation_loss_l1, validation_loss_NMSE, validation_loss_PSNR, validation_loss_SSIM = evaluate_loss_dataloader(model, val_loader)
        val_loss_history.append(validation_loss_NMSE)

        # init the min value and index
        min_value = val_loss_history[0] 
        min_index = 0     
        for i in range(1, len(val_loss_history)):
            if val_loss_history[i] < min_value:
                min_value = val_loss_history[i]
                min_index = i 

    return min_index

def average_early_stopping_epoch(model, checkpoint_path, optimizer, dataset, training_epoch=30, split_ratio=[7,3], repeat_times=5): 
    """
    The function is just repeat the early-stopping epoch calculation several times and output the average

    Parameters:

    """
    min_index_history = []
    for i in range(repeat_times):
        min_index = cal_early_stopping(model, checkpoint_path, dataset, optimizer, training_epoch, split_ratio)
        min_index_history.append(min_index)

    min_index_mean = sum(min_index_history) / len(min_index_history) 
    suggest_epoch = round(min_index_mean)+1
    return suggest_epoch




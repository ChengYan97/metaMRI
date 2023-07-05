import torch
import time

from functions.training.losses import SSIMLoss
from functions.math import complex_abs

def evaluate2c_imagepair(model, input_image, target_image, mean, std):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model.eval()
    l1_loss = torch.nn.L1Loss(reduction='sum')
    ssim_fct = SSIMLoss()
    psner_mse_fct = torch.nn.MSELoss(reduction='mean')
    mse_fct = torch.nn.MSELoss(reduction='sum')

    input_image = input_image.to(device)
    target_image = target_image.to(device)
    std = std.to(device)
    mean = mean.to(device)

    # time start
    start_time = time.time()
    output_image = model(input_image.unsqueeze(0))
    output_image = output_image * std + mean

    # Move complex dim to end, apply complex abs, insert channel dimension
    output_image_1c = complex_abs(torch.moveaxis(output_image , 1, -1 ))

    # time end
    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Inference time: ', elapsed_time)
    # NMAE
    nmae_loss = (l1_loss(output_image_1c, target_image) / torch.sum(torch.abs(target_image))).item()
    # NMSE 
    nmse_loss = (mse_fct(output_image_1c, target_image) / torch.sum(torch.abs(target_image)**2)).item()
    # PSNR
    psnr_loss = (20*torch.log10(torch.tensor(target_image.max().unsqueeze(0).item()))-10*torch.log10(psner_mse_fct(output_image_1c, target_image))).item()
    # SSIM = 1 - loss
    ssim_loss = 1 - ssim_fct(output_image_1c, target_image, data_range = target_image.max().unsqueeze(0)).item()

    return nmae_loss, nmse_loss, psnr_loss, ssim_loss, output_image, output_image_1c



def evaluate_validation_loss(model, dataloader):
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
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

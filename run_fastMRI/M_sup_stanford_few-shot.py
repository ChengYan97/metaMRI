import random
import numpy as np
import pickle
import h5py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from functions.data.transforms import rss_torch, to_tensor, complex_center_crop, rss_complex, normalize_separate_over_ch
from functions.fftc import fft2c_new as fft2c
from functions.fftc import ifft2c_new as ifft2c
from functions.math import complex_abs, complex_mul, complex_conj
from functions.models.unet import Unet
from functions.data.subsample import create_mask_for_mask_type
from functions.training.losses import SSIMLoss
from functions.helper import average_early_stopping_epoch, evaluate_loss_dataloader, adapt

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

SEED = 5
INIT = 'maml'
TRAINING_EPOCH = 70
LR = 0.001
CA_LR = 0.0001
adapt_shot = 10

experiment_name = "E_sup_" + INIT + "_" + str(adapt_shot) + "few-adapt_lr" + str(LR) + "_stanford_seed" + str(SEED)
print('Experiment: ', experiment_name)
# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

# seed
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

### data path ###
mypath = '/media/hdd1/stanford_fastmri_format/'

### data ###
with open(mypath+'train_val_filenames','rb') as fn:
    [train_slices,test_slices] = pickle.load(fn)

### mask ###
with open('/cheng/metaMRI/ttt_for_deep_learning_cs/unet/train_data/stanford_mask','rb') as fn:
    mask2d = pickle.load(fn)
mask = torch.tensor(mask2d[0]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
mask = mask.to(device)

with open('/cheng/metaMRI/ttt_for_deep_learning_cs/unet/test_data/dataset_shift/mask2d','rb') as fn:
    mask2d_ = pickle.load(fn)
mask_test = torch.tensor(mask2d_[0]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
mask_test = mask_test.to(device)

### model ###
if INIT == 'standard':
    checkpoint_path = "/cheng/metaMRI/metaMRI/thesis/metaMRI/setup_8knee/checkpoints/E6.6+_standard(NMSE-lrAnneal)_T8x200_120epoch/E6.6+_standard(NMSE-lrAnneal)_T8x200_120epoch_E64_best.pth"
elif INIT == 'maml':
    checkpoint_path = "/cheng/metaMRI/metaMRI/thesis/metaMRI/setup_8knee/checkpoints/E6.4_maml(NMSE-lr-in1e-3-out1e-4)_T8x200_200epoch_E200_best.pth"
elif INIT == 'maml_12mix':
    checkpoint_path = '/cheng/metaMRI/metaMRI/save/E_MAML(NMSE-out-3-in-4)_T12x200mix_300epoch/E_MAML(NMSE-out-3-in-4)_T12x200mix_300epoch_E300.pth'
elif INIT == 'standard_12mix':
    checkpoint_path = '/cheng/metaMRI/metaMRI/save/E_standard(NMSE-lr1e-3CA4)_T12x200mix_120epoch/E_standard(NMSE-lr1e-3CA4)_T12x200mix_120epoch_E69_best.pth'
else: 
    print('Choose the initialization weight. ')

model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)

### optimizer ###
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, TRAINING_EPOCH/1, eta_min=CA_LR, last_epoch=-1)

def stanford_data_adapt(model, fewshot_train, optimizer): 
    random.shuffle(fewshot_train)   # shuffle
    model.train()
    train_loss = 0.0
    mse_fct = torch.nn.MSELoss(reduction='sum')
    for i, slice_file_name in enumerate(fewshot_train):
        slice = int(slice_file_name.split('-')[0])  # slice: 69
        file_name = slice_file_name.split('-')[-1]  # file_name: 'ge9.h5'
        #filename, file_extension = os.path.splitext(file_name)  # filename: 'ge9'. file_extension: '.h5'
        ### load the training sample
        f = h5py.File(mypath + file_name, 'r')
        kspace = f['kspace'][slice] # ground truth k-space: [8,320,320]
        kspace = to_tensor(kspace).to(device)   # complex k-space
        # ground truth image: [1,320,320]
        target_image = f['reconstruction_rss'][slice] 
        target_image = torch.tensor(target_image).unsqueeze(0).to(device)
        # under sampled image: [1,320,320]
        masked_kspace = kspace * mask + 0.0 # [8,320,320]
        crop_size = (target_image.shape[-2], target_image.shape[-1])
        input_image = rss_complex(complex_center_crop(ifft2c(masked_kspace), crop_size)).unsqueeze(0)
        # normalize input to have zero mean and std one
        input_image, mean, std = normalize_separate_over_ch(input_image, eps=1e-11)
        
        # training
        train_output = model(input_image.unsqueeze(0)).squeeze(0)
        train_output = train_output * std + mean

        loss = mse_fct(train_output, target_image) / torch.sum(torch.abs(target_image)**2)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    avg_train_loss = train_loss / len(fewshot_train)
    return avg_train_loss

def evaluate_loss_stanford_data(model, data_val):
    model.eval()
    total_ssim_loss = 0.0
    total_psnr_loss = 0.0
    total_nmse_loss = 0.0
    l1_loss = torch.nn.L1Loss(reduction='sum')
    ssim_fct = SSIMLoss()
    psner_mse_fct = torch.nn.MSELoss(reduction='mean')
    mse_fct = torch.nn.MSELoss(reduction='sum')

    for i, slice_file_name in enumerate(data_val):
        slice = int(slice_file_name.split('-')[0])  # slice: 69
        file_name = slice_file_name.split('-')[-1]  # file_name: 'ge9.h5'
        #filename, file_extension = os.path.splitext(file_name)  # filename: 'ge9'. file_extension: '.h5'
        ### load the training sample
        f = h5py.File(mypath + file_name, 'r')
        kspace = f['kspace'][slice] # ground truth k-space: [8,320,320]
        kspace = to_tensor(kspace).to(device)   # complex k-space
        # ground truth image: [1,320,320]
        target_image = f['reconstruction_rss'][slice]   # [320,320]
        target_image = torch.tensor(target_image).unsqueeze(0).to(device)
        # under sampled image: [1,320,320]
        masked_kspace = kspace * mask_test + 0.0 # [8,320,320]
        crop_size = (target_image.shape[-2], target_image.shape[-1])
        input_image = rss_complex(complex_center_crop(ifft2c(masked_kspace), crop_size)).unsqueeze(0)
        # normalize input to have zero mean and std one
        input_image, mean, std = normalize_separate_over_ch(input_image, eps=1e-11)
        
        # testing
        val_output = model(input_image.unsqueeze(0)).squeeze(0)
        val_output = val_output * std + mean

        # NMSE 
        nmse_loss = mse_fct(val_output, target_image) / torch.sum(torch.abs(target_image)**2)
        total_nmse_loss += nmse_loss.item()
        # PSNR
        psnr_loss = 20*torch.log10(torch.tensor(target_image.max().unsqueeze(0).item()))-10*torch.log10(psner_mse_fct(val_output,target_image))
        total_psnr_loss += psnr_loss.item()
        # SSIM = 1 - loss
        ssim_loss = ssim_fct(val_output.unsqueeze(0), target_image.unsqueeze(0), data_range = target_image.max().unsqueeze(0))
        total_ssim_loss += (1-ssim_loss.item())
    validation_loss_NMSE = total_nmse_loss / len(data_val)
    validation_loss_PSNR = total_psnr_loss / len(data_val)
    validation_loss_SSIM = total_ssim_loss / len(data_val)

    return validation_loss_NMSE, validation_loss_PSNR, validation_loss_SSIM

########## training #########

test_loss_NMSE, test_loss_PSNR, test_loss_SSIM = evaluate_loss_stanford_data(model, test_slices)
writer.add_scalar("Testing NMSE", test_loss_NMSE, 0)
writer.add_scalar("Testing PSNR", test_loss_PSNR, 0)
writer.add_scalar("Testing SSIM", test_loss_SSIM, 0)
# sample few-shot
indices = random.sample(range(len(train_slices)), adapt_shot)
fewshot_train = [train_slices[i] for i in indices]

for iteration in range(TRAINING_EPOCH):
    print('Iteration: ', iteration+1)
    # training
    training_loss = stanford_data_adapt(model, fewshot_train, optimizer)
    writer.add_scalar("Adaptation training NMSE", training_loss, iteration+1)

    # testing
    test_loss_NMSE, test_loss_PSNR, test_loss_SSIM = evaluate_loss_stanford_data(model, test_slices)
    writer.add_scalar("Testing NMSE", test_loss_NMSE, iteration+1)
    writer.add_scalar("Testing PSNR", test_loss_PSNR, iteration+1)
    writer.add_scalar("Testing SSIM", test_loss_SSIM, iteration+1)

    # cosine annealing
    scheduler.step()

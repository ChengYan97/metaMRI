import time
import random
import numpy as np
import pandas as pd
import sys
import pickle
import h5py
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import learn2learn as l2l
from tqdm import tqdm
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from functions.fftc import fft2c_new as fft2c
from functions.fftc import ifft2c_new as ifft2c
from functions.math import complex_abs, complex_mul, complex_conj
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform_TTTpaper, center_crop, scale, normalize_separate_over_ch
# Import a torch.utils.data.Dataset class that takes a list of data examples, a path to those examples
# a data transform and outputs a torch dataset.
from functions.data.mri_dataset import SliceDataset
# Unet architecture as nn.Module
from functions.models.unet import Unet
# Function that returns a MaskFunc object either for generatig random or equispaced masks
from functions.data.subsample import create_mask_for_mask_type
# Implementation of SSIMLoss
from functions.training.losses import SSIMLoss



plt.rcParams.update({"text.usetex": True, "font.family": "serif", "font.serif": ["Computer Modern Roman"]})

colors = ['b','r','k','g','m','c','tab:brown','tab:orange','tab:pink','tab:gray','tab:olive','tab:purple']

markers = ["v","o","^","1","*",">","d","<","s","P","X"]
FONTSIZE = 22

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

# seed
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)


# data path
# path_test = '/cheng/metaMRI/metaMRI/data_dict/E11.1/Q/brain_test_AXT1POST_Skyra_5-8.yaml'
# path_test_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/E11.1/Q/sensmap_test/'
path_test = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_test_100.yaml'
path_test_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_test/'

# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform = UnetDataTransform_TTTpaper('multicoil', mask_func = mask_function, use_seed=True, mode='adapt')

# training dataset and data loader
testset = SliceDataset(dataset = path_test, path_to_dataset='', 
                path_to_sensmaps = path_test_sensmaps, provide_senmaps=True, 
                challenge="multicoil", transform = data_transform, use_dataset_cache=True)

test_dataloader = torch.utils.data.DataLoader(dataset = testset, batch_size = 1, 
                shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)


checkpoint_path = '/cheng/metaMRI/metaMRI/save/E11.7_sup(l1_CA-1e-3-4_Q)_T300_150epoch/E11.7_sup(l1_CA-1e-3-4_Q)_T300_150epoch_E84_best.pth'
# '/cheng/metaMRI/metaMRI/save/E11.7_sup(l1_CA-1e-3-4_P)_T300_150epoch/E11.7_sup(l1_CA-1e-3-4_P)_T300_150epoch_E85_best.pth'
model = Unet(in_chans=2, out_chans=2, chans=32, num_pool_layers=4, drop_prob=0.0)
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)

def evaluate(model, k_input, k_gt, conj_sensmaps): 
    # k_input, k_gt: [1, coils, height, width, channel]

    ssim_fct = SSIMLoss()
    l1_loss = torch.nn.L1Loss(reduction='sum')
    # gt image: complex
    ground_truth_image = ifft2c(k_gt.squeeze(0))
    ground_truth_image = complex_mul(ground_truth_image.squeeze(0), conj_sensmaps.squeeze(0))
    ground_truth_image = ground_truth_image.sum(dim=0, keepdim=False)
    ground_truth_image = torch.moveaxis( ground_truth_image , -1, 0 ) 

    # A†y
    train_inputs = ifft2c(k_input.squeeze(0)) #shape: coils,height,width,2
    train_inputs = complex_mul(train_inputs, conj_sensmaps.squeeze(0))
    train_inputs = train_inputs.sum(dim=0, keepdim=False) #shape: height,width,2
    train_inputs = torch.moveaxis( train_inputs , -1, 0 ) # move complex channels to channel dimension
    # normalize input to have zero mean and std one
    train_inputs, mean, std = normalize_separate_over_ch(train_inputs, eps=1e-11)

    # fθ(A†y)
    model_output = model(train_inputs.unsqueeze(0))
    model_output = model_output.squeeze(0) * std + mean

    # supervised loss [x, fθ(A†y)]
    # L1
    loss_l1 = (l1_loss(model_output, ground_truth_image) / torch.sum(torch.abs(ground_truth_image))).item()
    # SSIM = 1 - loss
    output_image_1c = complex_abs(torch.moveaxis(model_output , 0, -1 )).unsqueeze(0)
    ground_truth_image_1c = complex_abs(torch.moveaxis(ground_truth_image , 0, -1 )).unsqueeze(0)
    # center crop
    min_size = min(ground_truth_image_1c.shape[-2:])
    crop = torch.Size([min_size, min_size])
    ground_truth_image_1c = center_crop( ground_truth_image_1c, crop )
    output_image_1c = center_crop( output_image_1c, crop )
    loss_ssim = 1 - ssim_fct(output_image_1c, ground_truth_image_1c, data_range = ground_truth_image_1c.max().unsqueeze(0)).item()
    
    return loss_l1, loss_ssim

best_loss_l1_history=[]
best_loss_l1_index_history=[]
best_loss_ssim_history=[]
best_loss_ssim_index_history=[]
ssim_fct = SSIMLoss()
l1_loss = torch.nn.L1Loss(reduction='sum')
adapt_lr = 0.0001
TTT_epoch = 500

for iter, batch in tqdm(enumerate(test_dataloader)): 
    input_kspace, input_mask, kspace, sens_maps, sens_maps_conj, binary_background_mask, fname, slice_num = batch
    input_kspace = input_kspace.to(device)
    input_mask = input_mask.to(device)
    kspace = kspace.to(device)
    sens_maps = sens_maps.to(device)
    sens_maps_conj = sens_maps_conj.to(device)

    # model re-init
    model.load_state_dict(torch.load(checkpoint_path))

    # each data point TTT
    optimizer = torch.optim.Adam(model.parameters(),lr=adapt_lr)
    
    loss_l1_history = []
    loss_ssim_history = []
    self_loss_history = []

    for iteration in range(TTT_epoch): 
        ####### training ######
        # scale normalization
        scale_factor = scale(input_kspace.squeeze(0), model)

        # A†y
        scaled_input_kspace = scale_factor * input_kspace.squeeze(0)
        train_inputs = ifft2c(scaled_input_kspace) #shape: coils,height,width,2
        train_inputs = complex_mul(train_inputs, sens_maps_conj.squeeze(0))
        train_inputs = train_inputs.sum(dim=0, keepdim=False) #shape: height,width,2
        train_inputs = torch.moveaxis( train_inputs , -1, 0 ) # move complex channels to channel dimension
        # normalize input to have zero mean and std one
        train_inputs, mean, std = normalize_separate_over_ch(train_inputs, eps=1e-11)

        # fθ(A†y)
        model_output = model(train_inputs.unsqueeze(0))
        model_output = model_output.squeeze(0) * std + mean
        model_output = torch.moveaxis(model_output.unsqueeze(0), 1, -1 )
        # S fθ(A†y)
        output_sens_image = complex_mul(model_output, sens_maps.squeeze(0))
        # FS fθ(A†y)
        Fimg = fft2c(output_sens_image)
        # MFS fθ(A†y) = A fθ(A†y)
        Fimg_forward = Fimg * input_mask.squeeze(0)
        # consistency loss [y, Afθ(A†y)]
        loss_self = l1_loss(Fimg_forward, scaled_input_kspace) / torch.sum(torch.abs(scaled_input_kspace))
    
        optimizer.zero_grad()
        loss_self.backward()
        optimizer.step()
        #train_loss += loss.item()
        #print('TTT loss: ',loss_self.item())
        self_loss_history.append(loss_self.item())


        ###### evaluation ######
        loss_l1, loss_ssim = evaluate(model, input_kspace, kspace, sens_maps_conj)

        loss_l1_history.append(loss_l1)
        loss_ssim_history.append(loss_ssim)


    # best results for an example
    best_loss_l1 = min(loss_l1_history)
    print('best L1: ', best_loss_l1)
    best_loss_l1_epoch = np.argmin(loss_l1_history)
    print('best L1 epoch: ', best_loss_l1_epoch)
    best_loss_ssim = max(loss_ssim_history)
    print('best SSIM:', best_loss_ssim)
    best_loss_ssim_epoch = np.argmax(loss_ssim_history)
    print('best SSIM epoch: ', best_loss_ssim_epoch)

    best_loss_l1_index_history.append(best_loss_l1_epoch)
    best_loss_ssim_index_history.append(best_loss_ssim_epoch)
    best_loss_l1_history.append(best_loss_l1)
    best_loss_ssim_history.append(best_loss_ssim)


print("Testing average L1 loss: ", sum(best_loss_l1_history) / len(best_loss_l1_history))
print("Testing average L1 loss epoch: ", sum(best_loss_l1_index_history) / len(best_loss_l1_index_history))
print("Testing average SSIM loss: ", sum(best_loss_ssim_history) / len(best_loss_ssim_history))
print("Testing average SSIM loss epoch: ", sum(best_loss_ssim_index_history) / len(best_loss_ssim_index_history))
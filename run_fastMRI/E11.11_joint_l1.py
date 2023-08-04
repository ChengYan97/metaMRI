#%%
import random
import numpy as np
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform_TTTpaper_fixMask, rss_torch, scale_rss
# Import a torch.utils.data.Dataset class that takes a list of data examples, a path to those examples
# a data transform and outputs a torch dataset.
from functions.data.mri_dataset import SliceDataset
# Unet architecture as nn.Module
from functions.models.unet import Unet
# Function that returns a MaskFunc object either for generatig random or equispaced masks
from functions.data.subsample import create_mask_for_mask_type
# Implementation of SSIMLoss
from functions.training.losses import SSIMLoss

from functions.fftc import fft2c_new as fft2c
from functions.fftc import ifft2c_new as ifft2c
from functions.math import complex_abs, complex_mul, complex_conj


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

LOSS = 'sup'      # 'sup', 'joint'
DOMAIN = 'P'        # 'P', 'Q'

experiment_name = 'testE11.11_' + LOSS + '(l1_1e-5)'+ DOMAIN +'_T300_150epoch'
# 'E11.10_joint(l1_CA-1e-3-4_Q)_T300_150epoch'
print('Experiment: ', experiment_name)


# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

# seed
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

# hyperparameter
TRAINING_EPOCH = 150
BATCH_SIZE = 1
LR = 1e-5

# data path
if DOMAIN == 'P': 
    path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_train.yaml'
    path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_train/'
    path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_val.yaml'
    path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_val/'
    path_mask = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/knee_mask'

elif DOMAIN == 'Q':
    path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_train.yaml'
    path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_train/'
    path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_val.yaml'
    path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_val/'
    path_mask = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/brain_mask'


# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform = UnetDataTransform_TTTpaper_fixMask('multicoil', mask_func = mask_function, use_seed=True)

# training dataset and data loader
trainset = SliceDataset(dataset = path_train, path_to_dataset='', 
                path_to_sensmaps = path_to_train_sensmaps, provide_senmaps=True, 
                challenge="multicoil", transform = data_transform, 
                use_dataset_cache=True)

train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = BATCH_SIZE,
                shuffle = True, generator = torch.Generator().manual_seed(SEED), pin_memory = True)
print("Training date number: ", len(train_dataloader.dataset))

# validation dataset and data loader
# validationset = SliceDataset(dataset = path_val, path_to_dataset='', 
#                 path_to_sensmaps = path_to_val_sensmaps, provide_senmaps=True, 
#                 challenge="multicoil", transform = data_transform, 
#                 use_dataset_cache=True)

# val_dataloader = torch.utils.data.DataLoader(dataset = validationset, batch_size = 1, 
#                 shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)
# print("Validation date number: ", len(val_dataloader.dataset))

#%%

def train(model, dataloader, optimizer, scales_list): 
    model.train()
    train_loss = 0.0
    with open(path_mask,'rb') as fn:
        mask2d = pickle.load(fn)
    mask = torch.tensor(mask2d[0]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
    mask = mask.to(device)   

    for iter, batch in tqdm(enumerate(dataloader)):
        kspace, sens_maps, sens_maps_conj, _, fname, slice_num = batch
        kspace = kspace.squeeze(0).to(device)
        sens_maps = sens_maps.squeeze(0).to(device)
        sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

        # input k space
        input_kspace = kspace * mask + 0.0
        input_kspace = input_kspace.to(device)

        # gt image: x
        target_image = ifft2c(scales_list[iter] * kspace)
        # rss combine
        target_image = rss_torch(target_image)
        # sensmap combine
        # target_image = complex_mul(target_image, sens_maps_conj).sum(dim=0, keepdim=False)

        # A†y
        scale_input_kspace = scales_list[iter] * input_kspace
        train_inputs = ifft2c(scale_input_kspace) #shape: coils,height,width,2
        train_inputs = rss_torch(train_inputs)
        # train_inputs = complex_mul(train_inputs, sens_maps_conj).sum(dim=0, keepdim=False) #shape: height,width,2
        train_inputs = torch.moveaxis( train_inputs , -1, 0 ) # move complex channels to channel dimension

        # fθ(A†y)
        train_outputs = model(train_inputs.unsqueeze(0))
        train_outputs = train_outputs.squeeze(0)
        
        # supervised loss [x, fθ(A†y)]
        # [1, 2, 768, 392] -> [1, 768, 392]
        train_outputs_1c = complex_abs(torch.moveaxis(train_outputs.squeeze(0), 0, -1 )).unsqueeze(0)
        train_targets_1c = complex_abs(target_image).unsqueeze(0)
        loss_sup = l1_loss(train_outputs_1c, train_targets_1c) / torch.sum(torch.abs(train_targets_1c))
        
        # self-supervised loss
        if LOSS == 'sup': 
            loss_self = 0
        elif LOSS == 'joint':
            # fθ(A†y)
            train_outputs = torch.moveaxis(train_outputs.unsqueeze(0), 1, -1 )
            # S fθ(A†y)
            output_sens_image = complex_mul(train_outputs, sens_maps)
            #output_sens_image = output_sens_image.sum(dim=0, keepdim=False)
            # FS fθ(A†y)
            Fimg = fft2c(output_sens_image)
            # MFS fθ(A†y) = A fθ(A†y)
            Fimg_forward = Fimg * mask
            # self-supervised loss [y, Afθ(A†y)]
            loss_self = l1_loss(Fimg_forward, scale_input_kspace) / torch.sum(torch.abs(scale_input_kspace))
        
        # loss
        loss = loss_sup + loss_self

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss




model = Unet(in_chans=2, out_chans=2, chans=64, num_pool_layers=4, drop_prob=0.0)
model = model.to(device)


##########################
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
#scheduler = CosineAnnealingLR(optimizer, TRAINING_EPOCH/1, eta_min=1e-4, last_epoch=-1)
l1_loss = torch.nn.L1Loss(reduction='sum')


with open(path_mask,'rb') as fn:
    mask2d = pickle.load(fn)
mask = torch.tensor(mask2d[0]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
mask = mask.to(device)

print('Compute the scale factor for entire training data: ')
scales_list = []
for iter, batch in tqdm(enumerate(train_dataloader)):
    kspace, sens_maps, sens_maps_conj, _, fname, slice_num = batch
    kspace = kspace.squeeze(0).to(device)
    sens_maps = sens_maps.squeeze(0).to(device)
    sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

    # input k space
    input_kspace = kspace * mask + 0.0
    input_kspace = input_kspace.to(device)

    # scale normalization
    scale_factor = scale_rss(input_kspace, model)
    scales_list.append(scale_factor)

print('Training: ')
best_loss = 10.000
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss = train(model, train_dataloader, optimizer, scales_list)
    print('Training normalized L1', training_loss) 
    writer.add_scalar("Training normalized L1", training_loss, iteration+1)
  
    # val
    # validation_loss = evaluate(model, val_dataloader)
    # print('Validation normalized L1', validation_loss) 
    # writer.add_scalar("Validation normalized L1", validation_loss, iteration+1)
    # if best_loss > validation_loss:
    #     best_loss = validation_loss
    #     save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '_best.pth'
    #     torch.save((model.state_dict()), save_path)
    #     print('Model saved to', save_path)
    # else:
    #     pass
    #scheduler.step()
    #print('Learning rate: ', optimizer.param_groups[0]['lr'])
    save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '_best.pth'
    torch.save((model.state_dict()), save_path)


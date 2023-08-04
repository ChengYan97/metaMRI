#%%
import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform_TTTpaper, scale, normalize_separate_over_ch
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

LOSS = 'joint'      # 'sup', 'joint'

experiment_name = 'E11.9_' + LOSS + '(l1_CA-1e-3-4_Q)_T300_150epoch'


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
LR = 1e-3

# data path
# path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_train.yaml'
# path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_train/'
# path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_val.yaml'
# path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_val/'

path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_train.yaml'
path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_train/'
path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_val.yaml'
path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_val/'


# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform = UnetDataTransform_TTTpaper('multicoil', mask_func = mask_function, use_seed=True, mode='adapt')

# training dataset and data loader
trainset = SliceDataset(dataset = path_train, path_to_dataset='', 
                path_to_sensmaps = path_to_train_sensmaps, provide_senmaps=True, 
                challenge="multicoil", transform = data_transform, 
                use_dataset_cache=True)

train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = BATCH_SIZE,
                shuffle = True, generator = torch.Generator().manual_seed(SEED), pin_memory = True)
print("Training date number: ", len(train_dataloader.dataset))

# validation dataset and data loader
validationset = SliceDataset(dataset = path_val, path_to_dataset='', 
                path_to_sensmaps = path_to_val_sensmaps, provide_senmaps=True, 
                challenge="multicoil", transform = data_transform, 
                use_dataset_cache=True)

val_dataloader = torch.utils.data.DataLoader(dataset = validationset, batch_size = 1, 
                shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)
print("Validation date number: ", len(val_dataloader.dataset))

#%%

def train(model, dataloader, optimizer): 
    model.train()
    train_loss = 0.0

    for iter, batch in tqdm(enumerate(dataloader)):
        input_kspace, input_mask, kspace, sens_maps, sens_maps_conj, binary_background_mask, fname, slice_num = batch
        input_kspace = input_kspace.to(device)
        input_mask = input_mask.to(device)
        kspace = kspace.to(device)
        sens_maps = sens_maps.to(device)
        sens_maps_conj = sens_maps_conj.to(device)

        # scale normalization
        scale_factor = scale(input_kspace.squeeze(0), model)

        # gt image: x
        target_image = ifft2c(scale_factor * kspace.squeeze(0))
        target_image = complex_mul(target_image.squeeze(0), sens_maps_conj.squeeze(0))
        target_image = target_image.sum(dim=0, keepdim=False)
        # ground truth image: no-complex to evaluate no center crop
        #ground_truth_image = complex_abs(target_image).unsqueeze(0)
        target_image = torch.moveaxis( target_image , -1, 0 ) 

        # A†y
        input_kspace = scale_factor * input_kspace.squeeze(0)
        train_inputs = ifft2c(input_kspace) #shape: coils,height,width,2
        train_inputs = complex_mul(train_inputs, sens_maps_conj.squeeze(0))
        train_inputs = train_inputs.sum(dim=0, keepdim=False) #shape: height,width,2
        train_inputs = torch.moveaxis( train_inputs , -1, 0 ) # move complex channels to channel dimension
        # normalize input to have zero mean and std one
        # train_inputs, mean, std = normalize_separate_over_ch(train_inputs, eps=1e-11)

        # fθ(A†y)
        train_outputs = model(train_inputs.unsqueeze(0))
        train_outputs = train_outputs.squeeze(0)# * std + mean
        
        # supervised loss [x, fθ(A†y)]
        # [1, 2, 768, 392] -> [1, 768, 392]
        train_outputs_1c = complex_abs(torch.moveaxis(train_outputs.squeeze(0), 0, -1 )).unsqueeze(0)
        train_targets_1c = complex_abs(torch.moveaxis(target_image.squeeze(0), 0, -1 )).unsqueeze(0)
        loss_sup = l1_loss(train_outputs_1c, train_targets_1c) / torch.sum(torch.abs(train_targets_1c))
        
        # self-supervised loss
        loss_self = 0
        if LOSS == 'joint':
            # fθ(A†y)
            train_outputs = torch.moveaxis(train_outputs.unsqueeze(0), 1, -1 )
            # S fθ(A†y)
            output_sens_image = complex_mul(train_outputs, sens_maps.squeeze(0))
            #output_sens_image = output_sens_image.sum(dim=0, keepdim=False)
            # FS fθ(A†y)
            Fimg = fft2c(output_sens_image)
            # MFS fθ(A†y) = A fθ(A†y)
            Fimg_forward = Fimg * input_mask.squeeze(0)
            # self-supervised loss [y, Afθ(A†y)]
            loss_self = l1_loss(Fimg_forward, input_kspace) / torch.sum(torch.abs(input_kspace))
        
        # loss
        loss = loss_sup + loss_self

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss


def evaluate(model, dataloader):
    model.eval()
    total_val_loss = 0.0
    
    for iter, batch in tqdm(enumerate(dataloader)): 
        input_kspace, input_mask, kspace, sens_maps, sens_maps_conj, binary_background_mask, fname, slice_num = batch
        input_kspace = input_kspace.to(device)
        input_mask = input_mask.to(device)
        kspace = kspace.to(device)
        sens_maps = sens_maps.to(device)
        sens_maps_conj = sens_maps_conj.to(device)

        # scale normalization
        scale_factor = scale(input_kspace.squeeze(0), model)

        # gt image: x
        target_image = ifft2c(scale_factor * kspace.squeeze(0))
        target_image = complex_mul(target_image.squeeze(0), sens_maps_conj.squeeze(0))
        target_image = target_image.sum(dim=0, keepdim=False)
        # ground truth image: no-complex to evaluate no center crop
        #ground_truth_image = complex_abs(target_image).unsqueeze(0)
        target_image = torch.moveaxis( target_image , -1, 0 ) 

        # A†y
        input_kspace = scale_factor * input_kspace.squeeze(0)
        train_inputs = ifft2c(input_kspace) #shape: coils,height,width,2
        train_inputs = complex_mul(train_inputs, sens_maps_conj.squeeze(0))
        train_inputs = train_inputs.sum(dim=0, keepdim=False) #shape: height,width,2
        train_inputs = torch.moveaxis( train_inputs , -1, 0 ) # move complex channels to channel dimension
        # normalize input to have zero mean and std one
        #train_inputs, mean, std = normalize_separate_over_ch(train_inputs, eps=1e-11)

        # fθ(A†y)
        train_outputs = model(train_inputs.unsqueeze(0))
        train_outputs = train_outputs.squeeze(0)# * std + mean
        
        # supervised loss [x, fθ(A†y)]
        train_outputs_1c = complex_abs(torch.moveaxis(train_outputs.squeeze(0), 0, -1 )).unsqueeze(0)
        train_targets_1c = complex_abs(torch.moveaxis(target_image.squeeze(0), 0, -1 )).unsqueeze(0)
        loss_sup = l1_loss(train_outputs_1c, train_targets_1c) / torch.sum(torch.abs(train_targets_1c))
        
        # self-supervised loss
        loss_self = 0
        if LOSS == 'joint':
            train_outputs = torch.moveaxis(train_outputs.unsqueeze(0), 1, -1 )
            output_sens_image = complex_mul(train_outputs, sens_maps.squeeze(0))
            Fimg = fft2c(output_sens_image)
            Fimg_forward = Fimg * input_mask.squeeze(0)
            loss_self = l1_loss(Fimg_forward, input_kspace) / torch.sum(torch.abs(input_kspace))
        
        # loss
        loss = loss_sup + loss_self
        total_val_loss += loss.item()
        
    validation_loss = total_val_loss / len(dataloader)
    return validation_loss


model = Unet(in_chans=2, out_chans=2, chans=32, num_pool_layers=4, drop_prob=0.0)
model = model.to(device)


##########################
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
scheduler = CosineAnnealingLR(optimizer, TRAINING_EPOCH/1, eta_min=1e-4, last_epoch=-1)
l1_loss = torch.nn.L1Loss(reduction='sum')

best_loss = 10.000
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss = train(model, train_dataloader, optimizer)
    print('Training normalized L1', training_loss) 
    writer.add_scalar("Training normalized L1", training_loss, iteration+1)
  
    # val
    validation_loss = evaluate(model, val_dataloader)
    print('Validation normalized L1', validation_loss) 
    writer.add_scalar("Validation normalized L1", validation_loss, iteration+1)
    if best_loss > validation_loss:
        best_loss = validation_loss
        save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '_best.pth'
        torch.save((model.state_dict()), save_path)
        print('Model saved to', save_path)
    else:
        pass
    scheduler.step()
    #print('Learning rate: ', optimizer.param_groups[0]['lr'])


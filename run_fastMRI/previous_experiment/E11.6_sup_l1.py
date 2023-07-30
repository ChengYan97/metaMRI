#%%
import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform_sens_TTT, complex_center_crop,center_crop_to_smallest
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


experiment_name = 'E11.6_sup(l1_1e-5_P)_T300_150epoch'

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
path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_train.yaml'
path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_train/'
path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_val.yaml'
path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_val/'

# path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_train.yaml'
# path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_train/'
# path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_val.yaml'
# path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_val/'


# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform_sens_TTT('multicoil', mask_func = mask_function, use_seed=False, mode='train')
data_transform = UnetDataTransform_sens_TTT('multicoil', mask_func = mask_function, use_seed=True, mode='adapt')

# training dataset and data loader
trainset = SliceDataset(dataset = path_train, path_to_dataset='', 
                path_to_sensmaps=path_to_train_sensmaps, provide_senmaps=True, 
                challenge="multicoil", 
                transform = data_transform_train, 
                use_dataset_cache=True)

train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = BATCH_SIZE,
                shuffle = True, generator = torch.Generator().manual_seed(SEED), pin_memory = True)
print("Training date number: ", len(train_dataloader.dataset))

# validation dataset and data loader
validationset = SliceDataset(dataset = path_val, path_to_dataset='', 
                path_to_sensmaps=path_to_val_sensmaps, provide_senmaps=True, 
                challenge="multicoil", 
                transform=data_transform, 
                use_dataset_cache=True)

val_dataloader = torch.utils.data.DataLoader(dataset = validationset, batch_size = 1, 
                shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = False)
print("Validation date number: ", len(val_dataloader.dataset))

#%%
def train(model, dataloader, optimizer): 
    model.train()
    train_loss = 0.0

    for iter, batch in tqdm(enumerate(dataloader)):
        input_image, target_image, _, mean, std, _, _, input_kspace, input_mask, _, _, sens_maps, binary_background_mask = batch
        train_inputs = input_image.to(device)
        train_targets = target_image.to(device)
        input_kspace = input_kspace.to(device)
        input_mask = input_mask.to(device)
        sens_maps = sens_maps.to(device)
        std = std.to(device)
        mean = mean.to(device)
        binary_background_mask = binary_background_mask.to(device)
        # fθ(A†y)
        train_outputs = model(train_inputs)
        train_outputs = train_outputs * std + mean
        
        # supervised loss [x, fθ(A†y)]
        loss_sup = l1_loss(train_outputs, train_targets) / torch.sum(torch.abs(train_targets))
    

        optimizer.zero_grad()
        loss_sup.backward()
        optimizer.step()
        train_loss += loss_sup.item()

    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss


def evaluate(model, dataloader):
    model.eval()
    total_val_loss = 0.0
    
    for iter, batch in tqdm(enumerate(dataloader)): 
        input_image, target_image, ground_truth_image, mean, std, fname, slice_num, input_kspace, input_mask, target_kspace, target_mask, sens_maps, binary_background_mask = batch
        val_inputs = input_image.to(device)
        target_image = target_image.to(device)
        std = std.to(device)
        mean = mean.to(device)
        binary_background_mask = binary_background_mask.to(device)

        val_outputs = model(val_inputs)
        val_outputs = val_outputs * std + mean
        # binary_background_mask
        val_outputs = val_outputs * binary_background_mask

        loss = l1_loss(val_outputs, target_image) / torch.sum(torch.abs(target_image))

        total_val_loss += loss.item()
        
    validation_loss = total_val_loss / len(dataloader)
    return validation_loss


model = Unet(in_chans=2, out_chans=2, chans=32, num_pool_layers=4, drop_prob=0.0)
model = model.to(device)


##########################
optimizer = torch.optim.Adam(model.parameters(),lr=LR)
#scheduler = CosineAnnealingLR(optimizer, TRAINING_EPOCH/1, eta_min=0.0001, last_epoch=-1)
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
    
    #scheduler.step()
    #print('Learning rate: ', optimizer.param_groups[0]['lr'])


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
from functions.data.transforms import UnetDataTransform_TTTpaper_fixMask, normalize_separate_over_ch, rss_torch
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
COIL = 'sensmap'   # 'rss', 'sensmap'
KSPACE = False
background_flippping = False

# hyperparameter
LR = 1e-5
TRAINING_EPOCH = 300
BATCH_SIZE = 1

experiment_name = 'E13.1_' + LOSS + '(l1_1e-5)'+ DOMAIN +'_T300_300epoch'
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


# data path
if DOMAIN == 'P': 
    path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_train_300.yaml'
    path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_train/'
    path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_val copy 5.yaml'
    path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_val/'
    path_mask = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/knee_mask'

elif DOMAIN == 'Q':
    path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_train_300.yaml'
    path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_train/'
    path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_val.yaml'
    path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_val/'
    path_mask = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/brain_mask'


# mask function and data transform
data_transform = UnetDataTransform_TTTpaper_fixMask('multicoil')

# training dataset and data loader
trainset = SliceDataset(dataset = path_train, path_to_dataset='', 
                path_to_sensmaps = path_to_train_sensmaps, provide_senmaps=True, 
                challenge="multicoil", transform = data_transform, 
                use_dataset_cache=True)

train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = BATCH_SIZE,
                shuffle = True, generator = torch.Generator().manual_seed(SEED), pin_memory = True)
print("Training date number: ", len(train_dataloader.dataset))

if LOSS == 'sup':
    valset = SliceDataset(dataset = path_val, path_to_dataset='', 
                    path_to_sensmaps = path_to_val_sensmaps, provide_senmaps=True, 
                    challenge="multicoil", transform = data_transform, 
                    use_dataset_cache=True)

    valset_dataloader = torch.utils.data.DataLoader(dataset = valset, batch_size = BATCH_SIZE,
                    shuffle = True, generator = torch.Generator().manual_seed(SEED), pin_memory = True)
    print("Validation date number: ", len(valset_dataloader.dataset))


#%%

def train(model, dataloader, optimizer, mask): 
    model.train()
    train_loss = 0.0

    for iter, batch in tqdm(enumerate(dataloader)):
        kspace, sens_maps, sens_maps_conj, binary_background_mask, fname, slice_num = batch
        kspace = kspace.squeeze(0).to(device)
        sens_maps = sens_maps.squeeze(0).to(device)
        sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

        # input k space
        input_kspace = kspace * mask + 0.0

        # x: gt image [1,height,width]
        if COIL == 'rss':
            target_image_1c = rss_torch(complex_abs(ifft2c(kspace))).unsqueeze(0)
        elif COIL == 'sensmap':
            target_image_1c = complex_abs(complex_mul(ifft2c(kspace), sens_maps_conj).sum(dim=0, keepdim=False)).unsqueeze(0)

        # A†y: [2, height, width]
        if COIL == 'rss':
            train_inputs = torch.moveaxis(rss_torch(ifft2c(input_kspace)), -1, 0 )
        elif COIL == 'sensmap':    
            train_inputs = torch.moveaxis(complex_mul(ifft2c(input_kspace), sens_maps_conj).sum(dim=0, keepdim=False), -1, 0 )

        # normalize input to have zero mean and std one
        train_inputs, mean, std = normalize_separate_over_ch(train_inputs, eps=1e-11)
        
        # fθ(A†y): 
        # [2, height, width]
        train_output_2c = model(train_inputs.unsqueeze(0)).squeeze(0) * std + mean
        # [height, width, 2]
        train_output_2c = torch.moveaxis(train_output_2c.unsqueeze(0), 1, -1 )
        # [1, height, width]
        train_outputs_1c = complex_abs(train_output_2c).unsqueeze(0)

        # S fθ(A†y)
        if background_flippping == True: 
            train_output_sens_image = torch.zeros(sens_maps.shape).to(device) 
            for j,s in enumerate(sens_maps):
                ss = s.clone()
                ss[torch.abs(ss)==0.0] = torch.abs(ss).max()    ####### background flipping
                train_output_sens_image[j,:,:,0] = train_output_2c[0,:,:,0] * ss[:,:,0] - train_output_2c[0,:,:,1] * ss[:,:,1]
                train_output_sens_image[j,:,:,1] = train_output_2c[0,:,:,0] * ss[:,:,1] + train_output_2c[0,:,:,1] * ss[:,:,0]
        else:     
            train_output_sens_image = complex_mul(train_output_2c, sens_maps)

        train_kspace_output = fft2c(train_output_sens_image)
        if KSPACE == True: 
            loss_sup = l1_loss(train_kspace_output, kspace) / torch.sum(torch.abs(kspace))
        else: 
            loss_sup = l1_loss(train_outputs_1c, target_image_1c) / torch.sum(torch.abs(target_image_1c))

        # self-supervised loss
        if LOSS == 'sup': 
            loss_self = 0
        elif LOSS == 'joint':
            # MFS fθ(A†y) = A fθ(A†y)
            kspace_forward = train_kspace_output * mask
            # self-supervised loss [y, Afθ(A†y)]
            loss_self = l1_loss(kspace_forward, input_kspace) / torch.sum(torch.abs(input_kspace))
        
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


print('Training: ')
best_loss = 10.000
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss = train(model, train_dataloader, optimizer, mask)
    print('Training normalized L1', training_loss) 
    writer.add_scalar("Training normalized L1", training_loss, iteration+1)

    #scheduler.step()
    #print('Learning rate: ', optimizer.param_groups[0]['lr'])
    save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '.pth'
    torch.save((model.state_dict()), save_path)

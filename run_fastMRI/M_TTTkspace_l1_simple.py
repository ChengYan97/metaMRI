#%%
import random
import numpy as np
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform_TTTpaper_fixMask, normalize_separate_over_ch, rss_torch, scale_sensmap, scale_sensmap
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
DOMAIN = 'Q'        # 'P', 'Q'
COIL = 'sensmap'   # 'sensmap'
background_flippping = False

experiment_name = 'E_tttkspace_' + COIL + '_' + LOSS + '(l1_1e-5)'+ DOMAIN +'_T300_300epoch'


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
TRAINING_EPOCH = 300
BATCH_SIZE = 1
LR = 1e-5

# data path
if DOMAIN == 'P': 
    path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_train.yaml'
    path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_train/'
    path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_val_match.yaml'
    path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_val/'
    path_mask = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/knee_mask'

elif DOMAIN == 'Q':
    path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_train.yaml'
    path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_train/'
    path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_val.yaml'
    path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_val/'
    path_mask = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/brain_mask'

# data transform
data_transform = UnetDataTransform_TTTpaper_fixMask('multicoil')

# training dataset and data loader
trainset = SliceDataset(dataset = path_train, path_to_dataset='', 
                path_to_sensmaps = path_to_train_sensmaps, provide_senmaps=True, 
                challenge="multicoil", transform = data_transform, 
                use_dataset_cache=True)

train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = BATCH_SIZE,
                shuffle = False, generator = torch.Generator().manual_seed(SEED), pin_memory = True)
print("Training date number: ", len(train_dataloader.dataset))

# val dataset and data loader
valset = SliceDataset(dataset = path_val, path_to_dataset='', 
                path_to_sensmaps = path_to_val_sensmaps, provide_senmaps=True, 
                challenge="multicoil", transform = data_transform, 
                use_dataset_cache=True)

valset_dataloader = torch.utils.data.DataLoader(dataset = valset, batch_size = BATCH_SIZE,
                shuffle = False, generator = torch.Generator().manual_seed(SEED), pin_memory = True)
print("Validation date number: ", len(valset_dataloader.dataset))


#%%

def train(model, dataloader, optimizer, mask): 
    model.train()
    train_loss = 0.0

    for iter, batch in tqdm(enumerate(dataloader)):
        kspace, sens_maps, sens_maps_conj, _, fname, slice_num = batch
        kspace = kspace.squeeze(0).to(device)
        sens_maps = sens_maps.squeeze(0).to(device)
        sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

        # input k space
        input_kspace = kspace * mask + 0.0


        if COIL == 'rss':
            train_inputs = rss_torch(ifft2c(input_kspace))
        elif COIL == 'sensmap':    
            train_inputs = complex_mul(ifft2c(input_kspace), sens_maps_conj).sum(dim=0, keepdim=False)
        # [height, width, 2]
        train_inputs = torch.moveaxis( train_inputs , -1, 0 ) # move complex channels to channel dimension
        # [2, height, width]
        # normalize input to have zero mean and std one
        train_inputs, mean, std = normalize_separate_over_ch(train_inputs, eps=1e-11)
        # fθ(A†y)
        train_outputs = model(train_inputs.unsqueeze(0)) # [1, 2, height, width]
        train_outputs = train_outputs.squeeze(0) * std + mean

        # supervised loss 
        # fθ(A†y)
        train_outputs = torch.moveaxis(train_outputs.unsqueeze(0), 1, -1 )  #[1, height, width, 2]
        # S fθ(A†y)
        if background_flippping == True: 
            train_output_sens_image = torch.zeros(sens_maps.shape).to(device) 
            for j,s in enumerate(sens_maps):
                ss = s.clone()
                ss[torch.abs(ss)==0.0] = torch.abs(ss).max()#######
                train_output_sens_image[j,:,:,0] = train_outputs[0,:,:,0] * ss[:,:,0] - train_outputs[0,:,:,1] * ss[:,:,1]
                train_output_sens_image[j,:,:,1] = train_outputs[0,:,:,0] * ss[:,:,1] + train_outputs[0,:,:,1] * ss[:,:,0]
        else:     
            train_output_sens_image = complex_mul(train_outputs, sens_maps)
        kspace_output = fft2c(train_output_sens_image)
        # supervised loss [x, fθ(A†y)]: one channel image domain in TTT paper
        loss_sup = l1_loss(kspace_output, kspace) / torch.sum(torch.abs(kspace))
        
        # self-supervised loss
        if LOSS == 'sup': 
            loss_self = 0
        elif LOSS == 'joint':
            # MFS fθ(A†y) = A fθ(A†y)
            Fimg_forward = kspace_output * mask
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
    val_loss = 0.0

    for iter, batch in tqdm(enumerate(dataloader)):
        kspace, sens_maps, sens_maps_conj, binary_background_mask, fname, slice_num = batch
        kspace = kspace.squeeze(0).to(device)
        sens_maps = sens_maps.squeeze(0).to(device)
        sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

        # input k space
        input_kspace = kspace * mask + 0.0

        if COIL == 'rss':
            #target_image_1c = rss_torch(complex_abs(ifft2c(kspace * scales_list[iter]))).unsqueeze(0)
            val_inputs = rss_torch(ifft2c(input_kspace))
        elif COIL == 'sensmap':
            #target_image_1c = complex_abs(complex_mul(ifft2c(kspace * scales_list[iter]), sens_maps_conj).sum(dim=0, keepdim=False)).unsqueeze(0)
            val_inputs = complex_mul(ifft2c(input_kspace), sens_maps_conj).sum(dim=0, keepdim=False)

        # [height, width, 2]
        val_inputs = torch.moveaxis( val_inputs , -1, 0 ) # move complex channels to channel dimension
        # [2, height, width]
        # normalize input to have zero mean and std one
        val_inputs, mean, std = normalize_separate_over_ch(val_inputs, eps=1e-11)

        # fθ(A†y)
        val_outputs = model(val_inputs.unsqueeze(0))
        val_outputs = val_outputs.squeeze(0) * std + mean
        #val_outputs_1c = complex_abs(torch.moveaxis(val_outputs.squeeze(0), 0, -1 )).unsqueeze(0) # [1, height, width]
        val_outputs = torch.moveaxis(val_outputs.unsqueeze(0), 1, -1 )  #[1, height, width, 2]
        # S fθ(A†y)
        if background_flippping == True: 
            val_output_sens_image = torch.zeros(sens_maps.shape).to(device) 
            for j,s in enumerate(sens_maps):
                ss = s.clone()
                ss[torch.abs(ss)==0.0] = torch.abs(ss).max()#######
                val_output_sens_image[j,:,:,0] = val_outputs[0,:,:,0] * ss[:,:,0] - val_outputs[0,:,:,1] * ss[:,:,1]
                val_output_sens_image[j,:,:,1] = val_outputs[0,:,:,0] * ss[:,:,1] + val_outputs[0,:,:,1] * ss[:,:,0]
        else:     
            val_output_sens_image = complex_mul(val_outputs, sens_maps)
        val_kspace_output = fft2c(val_output_sens_image)
        # supervised loss [x, fθ(A†y)]: one channel image domain in TTT paper
        loss_val = l1_loss(val_kspace_output, kspace) / torch.sum(torch.abs(kspace))
        val_loss += loss_val.item()

    avg_val_loss = val_loss / len(dataloader)
    return avg_val_loss

model = Unet(in_chans=2, out_chans=2, chans=64, num_pool_layers=4, drop_prob=0.0)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=LR)
#scheduler = CosineAnnealingLR(optimizer, TRAINING_EPOCH/1, eta_min=1e-4, last_epoch=-1)
l1_loss = torch.nn.L1Loss(reduction='sum')


with open(path_mask,'rb') as fn:
    mask2d = pickle.load(fn)
mask = torch.tensor(mask2d[0]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
mask = mask.to(device)


print('Training: ')
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss = train(model, train_dataloader, optimizer, mask)
    print('Training normalized L1', training_loss) 
    writer.add_scalar("Training normalized L1", training_loss, iteration+1)

    # val
    validation_loss = evaluate(model, valset_dataloader)
    print('Validation normalized L1', validation_loss) 
    writer.add_scalar("Validation normalized L1", validation_loss, iteration+1)
  
    save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '.pth'
    torch.save((model.state_dict()), save_path)

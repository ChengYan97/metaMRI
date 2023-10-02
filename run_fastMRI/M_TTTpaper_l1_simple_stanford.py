#%% 
import random
import numpy as np
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import learn2learn as l2l
from tqdm import tqdm
import h5py
from torch.utils.tensorboard import SummaryWriter
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import to_tensor, normalize_separate_over_ch
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
DOMAIN = 'P'        # 'P', 'Q'
COIL = 'sensmap'    # 'rss', 'sensmap'
background_flippping = False
# 0-1 normalization

#experiment_name = 'E_' + COIL + '_' + LOSS + '(l1_1e-5)'+ DOMAIN +'_T300_300epoch_simple'
experiment_name = 'testtttt'

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

### data path ###
mypath = '/media/hdd1/stanford_fastmri_format/'
sensmap_path = '/cheng/metaMRI/metaMRI/data_dict_temp/stanford_sensmap/'
### data ###
with open('/cheng/metaMRI/ttt_for_deep_learning_cs/unet/train_data/stanford_train','rb') as fn:
    train_slices = pickle.load(fn)

with open('/cheng/metaMRI/ttt_for_deep_learning_cs/unet/train_data/stanford_mask','rb') as fn:
    mask2d = pickle.load(fn)
mask = torch.tensor(mask2d[0]).unsqueeze(0).unsqueeze(0).unsqueeze(-1)
mask = mask.to(device)

# train_slices = [train_slices[i] for i in range(len(train_slices))]

#%%

def train(model, dataloader, optimizer): 
    model.train()
    train_loss = 0.0
    train_loss_sup = 0.0
    train_loss_self = 0.0
    for iter, slice_file_name in tqdm(enumerate(dataloader)):
        slice = slice_file_name['slice']  # slice: 69
        file_name = slice_file_name['filename']  # file_name: 'ge9.h5'
        filename_no_extension, file_extension = os.path.splitext(file_name)  # filename: 'ge9'. file_extension: '.h5'
        ### load the training k-space
        f = h5py.File(mypath + file_name, 'r')
        kspace = f['kspace'][slice] # ground truth k-space: [8,320,320]
        kspace = to_tensor(kspace).to(device)   # complex k-space [8,320,320,2]

        ### load the sensmap
        # your name style and path
        smap_fname = 'stanford_sensmap' + str(filename_no_extension) + '_smaps_slice' + str(slice) + '.h5'
        with h5py.File(sensmap_path + smap_fname, "r") as hf:
            sens_maps = hf["sens_maps"][()] #np.array of shape coils,height,width with complex valued entries
        sens_maps = to_tensor(sens_maps)
        sens_maps_conj = complex_conj(sens_maps)
        sens_maps = sens_maps.to(device)
        sens_maps_conj = sens_maps_conj.to(device)

        # input k space
        input_kspace = kspace * mask + 0.0

        # gt image: x and coarse reconstruction A†y
        if COIL == 'rss':
            target_image_1c = rss_torch(complex_abs(ifft2c(kspace))).unsqueeze(0)
            train_inputs = rss_torch(ifft2c(input_kspace))
        elif COIL == 'sensmap':
            target_image_1c = complex_abs(complex_mul(ifft2c(kspace), sens_maps_conj).sum(dim=0, keepdim=False)).unsqueeze(0)
            train_inputs = complex_mul(ifft2c(input_kspace), sens_maps_conj).sum(dim=0, keepdim=False)

        # [height, width, 2]
        train_inputs = torch.moveaxis( train_inputs , -1, 0 ) # move complex channels to channel dimension
        # [2, height, width]
        # normalize input to have zero mean and std one
        train_inputs, mean, std = normalize_separate_over_ch(train_inputs, eps=1e-11)
 
        # fθ(A†y)
        train_outputs = model(train_inputs.unsqueeze(0)) # [1, 2, height, width]
        train_outputs = train_outputs.squeeze(0) * std + mean
        train_outputs_1c = complex_abs(torch.moveaxis(train_outputs.squeeze(0), 0, -1 )).unsqueeze(0) # [1, height, width]

        # supervised loss [x, fθ(A†y)]: one channel image domain in TTT paper
        loss_sup = l1_loss(train_outputs_1c, target_image_1c) / torch.sum(torch.abs(target_image_1c))
        
        # self-supervised loss
        if LOSS == 'sup': 
            loss_self = torch.tensor(0.0)
        elif LOSS == 'joint':
            # fθ(A†y)
            train_outputs = torch.moveaxis(train_outputs.unsqueeze(0), 1, -1)    #[1, height, width, 2]
            # S fθ(A†y)
            if background_flippping == True: 
                output_sens_image = torch.zeros(sens_maps.shape).to(device) 
                for j,s in enumerate(sens_maps):
                    ss = s.clone()
                    ss[torch.abs(ss)==0.0] = torch.abs(ss).max()#######
                    output_sens_image[j,:,:,0] = train_outputs[0,:,:,0] * ss[:,:,0] - train_outputs[0,:,:,1] * ss[:,:,1]
                    output_sens_image[j,:,:,1] = train_outputs[0,:,:,0] * ss[:,:,1] + train_outputs[0,:,:,1] * ss[:,:,0]
            else: 
                output_sens_image = complex_mul(train_outputs, sens_maps)
        
            # FS fθ(A†y)
            Fimg = fft2c(output_sens_image)            # FS fθ(A†y)
            # MFS fθ(A†y) = A fθ(A†y)
            Fimg_forward = Fimg * mask
            # self-supervised loss [y, Afθ(A†y)]
            loss_self = l1_loss(Fimg_forward, input_kspace) / torch.sum(torch.abs(input_kspace))
        
        # loss
        loss = loss_sup + loss_self

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        train_loss_sup += loss_sup.item()
        train_loss_self += loss_self.item()

    avg_train_loss = train_loss / len(dataloader)
    avg_train_loss_sup = train_loss_sup / len(dataloader)
    avg_train_loss_self = train_loss_self / len(dataloader)
    return avg_train_loss, avg_train_loss_sup, avg_train_loss_self

# def evaluate(model, dataloader): 
#     model.eval()
#     val_loss = 0.0

#     for iter, batch in tqdm(enumerate(dataloader)):
#         kspace, sens_maps, sens_maps_conj, binary_background_mask, fname, slice_num = batch
#         kspace = kspace.squeeze(0).to(device)
#         sens_maps = sens_maps.squeeze(0).to(device)
#         sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

#         # input k space
#         input_kspace = kspace * mask + 0.0

#         if COIL == 'rss':
#             target_image_1c = rss_torch(complex_abs(ifft2c(kspace))).unsqueeze(0)
#             val_inputs = rss_torch(ifft2c(input_kspace))
#         elif COIL == 'sensmap':
#             target_image_1c = complex_abs(complex_mul(ifft2c(kspace), sens_maps_conj).sum(dim=0, keepdim=False)).unsqueeze(0)
#             val_inputs = complex_mul(ifft2c(input_kspace), sens_maps_conj).sum(dim=0, keepdim=False)

#         # [height, width, 2]
#         val_inputs = torch.moveaxis( val_inputs , -1, 0 ) # move complex channels to channel dimension
#         # [2, height, width]
#         # normalize input to have zero mean and std one
#         val_inputs, mean, std = normalize_separate_over_ch(val_inputs, eps=1e-11)
        
#         # fθ(A†y)
#         val_outputs = model(val_inputs.unsqueeze(0))
#         val_outputs = val_outputs.squeeze(0) * std + mean
#         val_outputs_1c = complex_abs(torch.moveaxis(val_outputs.squeeze(0), 0, -1 )).unsqueeze(0) # [1, height, width]

#         # supervised loss [x, fθ(A†y)]: one channel image domain in TTT paper
#         loss_val = l1_loss(val_outputs_1c, target_image_1c) / torch.sum(torch.abs(target_image_1c))

#         val_loss += loss_val.item()

#     avg_val_loss = val_loss / len(dataloader)
#     return avg_val_loss

model = Unet(in_chans=2, out_chans=2, chans=64, num_pool_layers=4, drop_prob=0.0)
model = model.to(device)

optimizer = torch.optim.Adam(model.parameters(),lr=LR)
#scheduler = CosineAnnealingLR(optimizer, TRAINING_EPOCH/1, eta_min=1e-4, last_epoch=-1)
l1_loss = torch.nn.L1Loss(reduction='sum')



print('Training: ')
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss, training_loss_sup, training_loss_self = train(model, train_slices, optimizer)
    print('Training normalized L1', training_loss) 
    writer.add_scalar("Training normalized L1", training_loss, iteration+1)
    writer.add_scalar("Training normalized L1 - sup loss", training_loss_sup, iteration+1)
    writer.add_scalar("Training normalized L1 - self loss", training_loss_self, iteration+1)
    
    # val
    # validation_loss = evaluate(model, valset_dataloader)
    # print('Validation normalized L1', validation_loss) 
    # writer.add_scalar("Validation normalized L1", validation_loss, iteration+1)

    save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '.pth'
    torch.save((model.state_dict()), save_path)

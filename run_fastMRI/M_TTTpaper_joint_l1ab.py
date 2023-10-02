#%% 
import random
import numpy as np
import pickle
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform_TTTpaper_fixMask, rss_torch, scale_rss, scale_sensmap, scale_sensmap
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
DOMAIN = 'P'        # 'P', 'Q'
COIL = 'sensmap'    # 'rss', 'sensmap'

Weight_LOSS = False
LAMBDA = 0.99           # λ sup + (1-λ) self

experiment_name = 'E_tttpaper_' + COIL + '_' + LOSS + '(l1_1e-5)'+ DOMAIN +'_T300_300epoch_RSSgtSENSErecon'
# experiment_name = 'E_tttpaper_' + COIL + '_SUPdominate_' + LOSS + '(l1_1e-5)'+ DOMAIN +'_T300_300epoch'
# experiment_name = 'E_tttpaper_' + COIL + '_balance_' + LOSS + '(l1_1e-5)'+ DOMAIN +'_T300_300epoch'

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
    path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_train_300.yaml'
    path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_train/'
    path_val = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_val_match.yaml'
    path_to_val_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_val/'
    path_mask = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/knee_mask'

elif DOMAIN == 'Q':
    path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_brain_train_300.yaml'
    path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_brain_train/'
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

def train(model, dataloader, optimizer, scales_list): 
    model.train()
    train_loss = 0.0
    train_loss_sup = 0.0
    train_loss_self = 0.0
    for iter, batch in tqdm(enumerate(dataloader)):
        kspace, sens_maps, sens_maps_conj, _, fname, slice_num = batch
        kspace = kspace.squeeze(0).to(device)
        sens_maps = sens_maps.squeeze(0).to(device)
        sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

        # input k space
        input_kspace = kspace * mask + 0.0

        # gt image: x
        target_image_1c = rss_torch(complex_abs(ifft2c(kspace * scales_list[iter]))).unsqueeze(0)


        # A†y
        scale_input_kspace = scales_list[iter] * input_kspace
        if COIL == 'rss':
            train_inputs = rss_torch(ifft2c(scale_input_kspace))
        elif COIL == 'sensmap':    
            train_inputs = complex_mul(ifft2c(scale_input_kspace), sens_maps_conj).sum(dim=0, keepdim=False)
        # [height, width, 2]
        train_inputs = torch.moveaxis( train_inputs , -1, 0 ) # move complex channels to channel dimension
        # [2, height, width]

        # fθ(A†y)
        train_outputs = model(train_inputs.unsqueeze(0)) # [1, 2, height, width]
        train_outputs_1c = complex_abs(torch.moveaxis(train_outputs.squeeze(0), 0, -1 )).unsqueeze(0) # [1, height, width]

        # supervised loss [x, fθ(A†y)]: one channel image domain in TTT paper
        loss_sup = l1_loss(train_outputs_1c, target_image_1c) / torch.sum(torch.abs(target_image_1c))
        
        # self-supervised loss
        if LOSS == 'sup': 
            loss_self = torch.tensor(0.0)
        elif LOSS == 'joint':
            # fθ(A†y)
            train_outputs = torch.moveaxis(train_outputs, 1, -1)    #[1, height, width, 2]
            # S fθ(A†y)
            output_sens_image = torch.zeros(sens_maps.shape).to(device) 
            for j,s in enumerate(sens_maps):
                ss = s.clone()
                ss[torch.abs(ss)==0.0] = torch.abs(ss).max()#######
                output_sens_image[j,:,:,0] = train_outputs[0,:,:,0] * ss[:,:,0] - train_outputs[0,:,:,1] * ss[:,:,1]
                output_sens_image[j,:,:,1] = train_outputs[0,:,:,0] * ss[:,:,1] + train_outputs[0,:,:,1] * ss[:,:,0]
            # FS fθ(A†y)
            Fimg = fft2c(output_sens_image)            # FS fθ(A†y)
            # MFS fθ(A†y) = A fθ(A†y)
            Fimg_forward = Fimg * mask
            # self-supervised loss [y, Afθ(A†y)]
            loss_self = l1_loss(Fimg_forward, scale_input_kspace) / torch.sum(torch.abs(scale_input_kspace))
        
        # loss
        if Weight_LOSS == True: 
            loss = LAMBDA*loss_sup + (1-LAMBDA)*loss_self
        else:
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


def evaluate(model, dataloader, scales_list): 
    model.eval()
    val_loss = 0.0

    for iter, batch in tqdm(enumerate(dataloader)):
        kspace, sens_maps, sens_maps_conj, binary_background_mask, fname, slice_num = batch
        kspace = kspace.squeeze(0).to(device)
        sens_maps = sens_maps.squeeze(0).to(device)
        sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

        # input k space
        input_kspace = kspace * mask + 0.0
        scale_input_kspace = scales_list[iter] * input_kspace

        target_image_1c = rss_torch(complex_abs(ifft2c(kspace * scales_list[iter]))).unsqueeze(0)
        val_inputs = complex_mul(ifft2c(scale_input_kspace), sens_maps_conj).sum(dim=0, keepdim=False)

        # [height, width, 2]
        val_inputs = torch.moveaxis( val_inputs , -1, 0 ) # move complex channels to channel dimension
        # [2, height, width]
        # normalize input to have zero mean and std one

        # fθ(A†y)
        val_outputs = model(val_inputs.unsqueeze(0))

        val_outputs_1c = complex_abs(torch.moveaxis(val_outputs.squeeze(0), 0, -1 )).unsqueeze(0) # [1, height, width]

        # supervised loss [x, fθ(A†y)]: one channel image domain in TTT paper
        loss_val = l1_loss(val_outputs_1c, target_image_1c) / torch.sum(torch.abs(target_image_1c))

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

print('Compute the scale factor for entire training data: ')
scales_list = []
for iter, batch in enumerate(train_dataloader):
    kspace, sens_maps, sens_maps_conj, _, fname, slice_num = batch
    kspace = kspace.squeeze(0).to(device)
    sens_maps = sens_maps.squeeze(0).to(device)
    sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

    # input k space
    input_kspace = kspace * mask + 0.0

    # scale normalization
    if COIL == 'rss':
        scale_factor = scale_rss(input_kspace, model)
    elif COIL == 'sensmap':
        scale_factor = scale_sensmap(input_kspace, model, sens_maps_conj)
    
    scales_list.append(scale_factor)
    print('{}/{} samples normalized.'.format(iter+1,len(train_dataloader)),'\r',end='')


print('Compute the scale factor for entire validation data: ')
scales_list_val = []
for iter, batch in enumerate(valset_dataloader):
    kspace, sens_maps, sens_maps_conj, _, fname, slice_num = batch
    kspace = kspace.squeeze(0).to(device)
    sens_maps = sens_maps.squeeze(0).to(device)
    sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

    # input k space
    input_kspace = kspace * mask + 0.0

    # scale normalization
    if COIL == 'rss':
        scale_factor = scale_rss(input_kspace, model)
    elif COIL == 'sensmap':
        scale_factor = scale_sensmap(input_kspace, model, sens_maps_conj)
    
    scales_list_val.append(scale_factor)
    print('{}/{} samples normalized.'.format(iter+1,len(valset_dataloader)),'\r',end='')


print('Training: ')
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss, training_loss_sup, training_loss_self = train(model, train_dataloader, optimizer, scales_list)
    print('Training normalized L1', training_loss) 
    writer.add_scalar("Training normalized L1", training_loss, iteration+1)
    writer.add_scalar("Training normalized L1 - sup loss", training_loss_sup, iteration+1)
    writer.add_scalar("Training normalized L1 - self loss", training_loss_self, iteration+1)

    # val
    validation_loss = evaluate(model, valset_dataloader, scales_list_val)
    print('Validation normalized L1', validation_loss) 
    writer.add_scalar("Validation normalized L1", validation_loss, iteration+1)
  
    save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '.pth'
    torch.save((model.state_dict()), save_path)

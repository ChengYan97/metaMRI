#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
import numpy as np
import copy
import pickle
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform_TTTpaper_fixMask, rss_torch, scale_sensmap
from functions.fftc import fft2c_new as fft2c
from functions.fftc import ifft2c_new as ifft2c
from functions.math import complex_abs, complex_mul, complex_conj
# Import a torch.utils.data.Dataset class that takes a list of data examples, a path to those examples
# a data transform and outputs a torch dataset.
from functions.data.mri_dataset import SliceDataset
# Unet architecture as nn.Module
from functions.models.unet import Unet
# Function that returns a MaskFunc object either for generatig random or equispaced masks
from functions.data.subsample import create_mask_for_mask_type
from functions.training.losses import SSIMLoss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

########################### experiment name ###########################
DOMAIN = 'P'        # 'P', 'Q'
COIL = 'sensmap'   # 'sensmap'

experiment_name = 'E_TTTmaml_in-sup_out-self(l1_out-5_in-5_AS-2)'+DOMAIN+'_T300_300epoch'


# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

# seed
SEED = 1
random.seed(SEED)
np.random.seed(SEED)
torch.cuda.manual_seed(SEED)
torch.manual_seed(SEED)

###########################  hyperparametes  ###########################
EPOCH = 300   
# enumalate the whole data once takes 180 outer loop
Inner_EPOCH = 1
BATCH_SIZE = 1
adapt_steps = 2
adapt_lr = 0.00001   # adapt θ': α
meta_lr = 0.00001    # update real model θ: β

###########################  data & dataloader  ###########################

# data path
if DOMAIN == 'P': 
    path_train = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/TTT_knee_train_300.yaml'
    path_to_train_sensmaps = '/cheng/metaMRI/metaMRI/data_dict/TTT_paper/sensmap_knee_train/'
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



#%% Check the data 
###########################  model  ###########################
# complex
model = Unet(in_chans = 2,out_chans = 2,chans = 64, num_pool_layers = 4,drop_prob = 0.0)
model = model.to(device)
maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)



###########################  MAML training  ###########################
optimizer = optim.Adam(maml.parameters(), meta_lr)
l1_loss = nn.L1Loss(reduction='sum')

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



### one training loop include 180 outer loop
for iter_ in range(EPOCH):    
    print('Iteration:', iter_+1)
    # update real model
    model.train()

    ###### 2: outer loop ######
    # here we consider 180 outer loop as one training loop
    meta_training_loss = 0.0
    meta_adaptation_loss_0 = 0.0
    meta_adaptation_loss_1 = 0.0

    ###### 3. Sample batch of tasks Ti ~ p(T) ######
    for iter, batch in enumerate(train_dataloader):
        kspace, sens_maps, sens_maps_conj, _, fname, slice_num = batch
        kspace = kspace.squeeze(0).to(device)
        sens_maps = sens_maps.squeeze(0).to(device)
        sens_maps_conj = sens_maps_conj.squeeze(0).to(device)

        # input k space
        input_kspace = kspace * mask + 0.0

        # gt kspace: y*
        target_image_1c = complex_abs(complex_mul(ifft2c(kspace * scales_list[iter]), sens_maps_conj).sum(dim=0, keepdim=False)).unsqueeze(0)

        # A†y
        scale_input_kspace = scales_list[iter] * input_kspace
        if COIL == 'rss':
            train_inputs = rss_torch(ifft2c(scale_input_kspace))
        elif COIL == 'sensmap':    
            train_inputs = complex_mul(ifft2c(scale_input_kspace), sens_maps_conj).sum(dim=0, keepdim=False)
        train_inputs = torch.moveaxis( train_inputs , -1, 0 ) # [2, height, width]


        total_update_loss = 0.0
        total_adapt_loss_0 = 0.0
        total_adapt_loss_1 = 0.0 
        ###### 4: inner loop ######
        # Ti only contain one task; one task is exactly 1 data point
        for inner_iter in range(Inner_EPOCH):
            # base learner
            learner = maml.clone()      #learner = torch.nn.DataParallel(learner, device_ids=[0,1,2,3])

            adapt_step_loss = []
            # adapt learner several step in a self-supervised manner
            for _ in range(adapt_steps): 
                ###### 5. Evaluate ∇θLTi(fθ) with respect to K examples ######
                adapt_output = learner(train_inputs.unsqueeze(0)) # [1, 2, height, width]
                adapt_output_1c = complex_abs(torch.moveaxis(adapt_output.squeeze(0), 0, -1 )).unsqueeze(0)
                adapt_loss = l1_loss(adapt_output_1c, target_image_1c) / torch.sum(torch.abs(target_image_1c))

                adapt_step_loss.append(adapt_loss.item())
                ###### 6. Compute  adapted  parameters  with  gradient  descent: θ′i = θ − α∇θLTi(fθ) ######
                learner.adapt(adapt_loss)
            
            ###### 7: inner loop end ######
            # for calculation efficient, some loss are still cumputed in this loop

            ####### 8. Update θ ← θ−β∇θ∑Ti∼p(T)LTi(fθ′i)   ######  
            # LTi(fθ′i)
            update_output = learner(train_inputs.unsqueeze(0))

            # update self-supervised loss
            # fθ(A†y)
            update_output = torch.moveaxis(update_output, 1, -1)    #[1, height, width, 2]
            # S fθ(A†y)
            output_sens_image = torch.zeros(sens_maps.shape).to(device) 
            for j,s in enumerate(sens_maps):
                ss = s.clone()
                ss[torch.abs(ss)==0.0] = torch.abs(ss).max()
                output_sens_image[j,:,:,0] = update_output[0,:,:,0] * ss[:,:,0] - update_output[0,:,:,1] * ss[:,:,1]
                output_sens_image[j,:,:,1] = update_output[0,:,:,0] * ss[:,:,1] + update_output[0,:,:,1] * ss[:,:,0]
            # FS fθ(A†y)
            Fimg = fft2c(output_sens_image)
            # MFS fθ(A†y) = A fθ(A†y)
            Fimg_forward = Fimg * mask
            # self-supervised loss [y, Afθ(A†y)]
            update_self_loss = l1_loss(Fimg_forward, scale_input_kspace) / torch.sum(torch.abs(scale_input_kspace))

            # joint loss
            update_loss = update_self_loss


            # ∑Ti∼p(T)LTi(fθ′i): Ti only contain one task
            total_update_loss += update_loss
            total_adapt_loss_0 += adapt_step_loss[0]
            total_adapt_loss_1 += adapt_step_loss[1]


        # del task_batch  # avoid cpu memory leak
        # del learner     # gpu

        # Update θ ← θ−β∇θ∑Ti∼p(T)LTi(fθ′i)
        optimizer.zero_grad()
        total_update_loss.backward()
        optimizer.step()
        # total_update_loss should be the meta training loss, 
        # but we want some value can evaluate the training through whole dataset
        # we use the mean of 180 outer loop loss as the meta training loss
        meta_training_loss += total_update_loss.item()
        meta_adaptation_loss_0 += total_adapt_loss_0
        meta_adaptation_loss_1 += total_adapt_loss_1


    
    writer.add_scalar("Meta 0-step Adaptation Loss (MAML)", meta_adaptation_loss_0/len(train_dataloader), iter_+1)
    writer.add_scalar("Meta 1-step Adaptation Loss (MAML)", meta_adaptation_loss_1/len(train_dataloader), iter_+1)


    print("Meta Training L1 (MAML)", meta_training_loss/len(train_dataloader))
    writer.add_scalar("Meta Training L1 (MAML)", meta_training_loss/len(train_dataloader), iter_+1)
    

    save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iter_+1) + '.pth'
    torch.save((model.state_dict()), save_path)
    
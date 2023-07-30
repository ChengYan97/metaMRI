#%%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import random
import numpy as np
import copy
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform_sens_TTT, scale
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
experiment_name = 'E11.7_maml(l1_CA-1e-3-4_Q)_T300_200epoch'

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
EPOCH = 200   
# enumalate the whole data once takes 180 outer loop
Inner_EPOCH = 1

K = 1      # the same examples for both inner loop and outer loop training
adapt_steps = 5
adapt_lr = 0.0001   # adapt θ': α
meta_lr = 0.001    # update real model θ: β

###########################  data & dataloader  ###########################

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

data_transform_train = UnetDataTransform_sens_TTT('multicoil', mask_func = mask_function, use_seed=False, mode='train')
data_transform = UnetDataTransform_sens_TTT('multicoil', mask_func = mask_function, use_seed=True, mode='adapt')

# training dataset and data loader
trainset = SliceDataset(dataset = path_train, path_to_dataset='', 
                path_to_sensmaps=path_to_train_sensmaps, provide_senmaps=True, 
                challenge="multicoil", 
                transform = data_transform_train, 
                use_dataset_cache=True)

train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = K,
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

#%% Check the data 
###########################  model  ###########################
# complex
model = Unet(in_chans = 2,out_chans = 2,chans = 32, num_pool_layers = 4,drop_prob = 0.0)
model = model.to(device)
maml = l2l.algorithms.MAML(model, lr=adapt_lr, first_order=False, allow_unused=True)

def evaluate(model, dataloader):
    model.eval()
    total_val_loss = 0.0
    
    for iter, batch in tqdm(enumerate(dataloader)): 
        input_image, target_image, _, mean, std, _, _, input_kspace, input_mask, _, _, sens_maps, binary_background_mask = batch
        input_image = input_image.to(device)
        target_image = target_image.to(device)
        input_kspace = input_kspace.to(device)
        input_mask = input_mask.to(device)
        sens_maps = sens_maps.to(device)
        std = std.to(device)
        mean = mean.to(device)
        binary_background_mask = binary_background_mask.to(device)

        # scale normalization
        scale_factor = scale(input_kspace.squeeze(0), model)
        input_kspace = scale_factor * input_kspace

        output = model(input_image)
        output = output * std + mean
        # supervised loss [x, fθ(A†y)]
        outputs_1c = complex_abs(torch.moveaxis(output.squeeze(0), 0, -1 )).unsqueeze(0)
        targets_1c = complex_abs(torch.moveaxis(target_image.squeeze(0), 0, -1 )).unsqueeze(0)
        loss_sup = l1_loss(outputs_1c, targets_1c) / torch.sum(torch.abs(targets_1c))
        
        # self-supervised loss[y, Afθ(A†y)]
        output = torch.moveaxis(output, 1, -1 )
        output_sens_image = complex_mul(output, sens_maps)
        Fimg = fft2c(output_sens_image)
        Fimg_forward = Fimg * input_mask
        loss_self = l1_loss(Fimg_forward, input_kspace) / torch.sum(torch.abs(input_kspace))
        
        # loss
        loss = loss_sup + loss_self
        total_val_loss += loss.item()
        
    validation_loss = total_val_loss / len(dataloader)
    return validation_loss


###########################  MAML training  ###########################
optimizer = optim.Adam(maml.parameters(), meta_lr)
scheduler = CosineAnnealingLR(optimizer, EPOCH/1, eta_min=0.0001, last_epoch=-1)
l1_loss = nn.L1Loss(reduction='sum')

best_loss = 10.000
### one training loop include 180 outer loop
for iter_ in range(EPOCH):    
    print('Iteration:', iter_+1)
    # update real model
    model.train()

    ###### 2: outer loop ######
    # here we consider 180 outer loop as one training loop
    meta_training_loss = 0.0
    meta_adaptation_loss = 0.0

    ###### 3. Sample batch of tasks Ti ~ p(T) ######
    for iter, batch in tqdm(enumerate(train_dataloader)):
        input_image, target_image, ground_truth_image, mean, std, fname, slice_num, input_kspace, input_mask, target_kspace, target_mask, sens_maps, binary_background_mask = batch        
        input_image = input_image.to(device)
        target_image = target_image.to(device)
        input_kspace = input_kspace.to(device)
        input_mask = input_mask.to(device)
        sens_maps = sens_maps.to(device)
        std = std.to(device)
        mean = mean.to(device)
        binary_background_mask = binary_background_mask.to(device)
        
        total_update_loss = 0.0
        total_adapt_loss = 0.0
        ###### 4: inner loop ######
        # Ti only contain one task; one task is exactly 1 data point
        for inner_iter in range(Inner_EPOCH):
            print('Inner loop: ', inner_iter+1)

            # base learner
            learner = maml.clone()      #learner = torch.nn.DataParallel(learner, device_ids=[0,1,2,3])
            adapt_kspace = input_kspace
            # adapt learner several step in a self-supervised manner
            for step in range(adapt_steps): 
                ###### 5. Evaluate ∇θLTi(fθ) with respect to K examples ######
                # self-supervised loss
                # scale normalization
                scale_factor = scale(adapt_kspace.squeeze(0), model)
                adapt_kspace = scale_factor * adapt_kspace
                # fθ(A†y)
                adapt_output = learner(input_image)
                adapt_output = adapt_output * std + mean
                adapt_output = torch.moveaxis(adapt_output, 1, -1 )
                # S fθ(A†y)
                output_sens_image = complex_mul(adapt_output, sens_maps)
                # FS fθ(A†y)
                Fimg = fft2c(output_sens_image)
                # MFS fθ(A†y) = A fθ(A†y)
                Fimg_forward = Fimg * input_mask
                # self-supervised loss [y, Afθ(A†y)] as adapt loss
                loss_self = l1_loss(Fimg_forward, adapt_kspace) / torch.sum(torch.abs(input_kspace))
                if fname == ['file_brain_AXT2_204_2040100']:
                    print("Record one example")
                    combined_iter = (iter_+1) * 10 + (step + 1) 
                    writer.add_scalar("Inner loop one example adapt L1 (MAML)", loss_self.item(), combined_iter)
                ###### 6. Compute  adapted  parameters  with  gradient  descent: θ′i = θ − α∇θLTi(fθ) ######
                learner.adapt(loss_self)
            
            ###### 7: inner loop end ######
            # for calculation efficient, some loss are still cumputed in this loop

            ####### 8. Update θ ← θ−β∇θ∑Ti∼p(T)LTi(fθ′i)   ######   
            # LTi(fθ′i)
            # scale normalization
            scale_factor = scale(input_kspace.squeeze(0), model)
            input_kspace = scale_factor * input_kspace
            # fθ(A†y)
            update_output = learner(input_image)
            update_output = update_output * std + mean
            # supervised loss [x, fθ(A†y)]
            # [1, 2, 768, 392] -> [1, 768, 392]
            update_outputs_1c = complex_abs(torch.moveaxis(update_output.squeeze(0), 0, -1 )).unsqueeze(0)
            update_targets_1c = complex_abs(torch.moveaxis(target_image.squeeze(0), 0, -1 )).unsqueeze(0)
            update_sup_loss = l1_loss(update_outputs_1c, update_targets_1c) / torch.sum(torch.abs(update_targets_1c))
            # update self-supervised loss
            update_output = torch.moveaxis(update_output, 1, -1 )
            # S fθ(A†y)
            update_output_sens_image = complex_mul(update_output, sens_maps)
            # FS fθ(A†y)
            update_Fimg = fft2c(update_output_sens_image)
            # MFS fθ(A†y) = A fθ(A†y)
            update_Fimg_forward = update_Fimg * input_mask
            # self-supervised loss [y, Afθ(A†y)] as adapt loss
            update_self_loss = l1_loss(update_Fimg_forward, input_kspace) / torch.sum(torch.abs(input_kspace))

            # joint loss
            update_loss = update_sup_loss + update_self_loss

            # ∑Ti∼p(T)LTi(fθ′i): Ti only contain one task
            total_update_loss += update_loss
            total_adapt_loss += loss_self.item()

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
        meta_adaptation_loss += total_adapt_loss

    print("Meta Adaptation L1 (MAML)", meta_adaptation_loss/len(train_dataloader))
    writer.add_scalar("Meta Adaptation L1 (MAML)", meta_adaptation_loss/len(train_dataloader), iter_+1)

    print("Meta Training L1 (MAML)", meta_training_loss/len(train_dataloader))
    writer.add_scalar("Meta Training L1 (MAML)", meta_training_loss/len(train_dataloader), iter_+1)
    
    # validate each epoch
    validation_loss = evaluate(model, val_dataloader)
    print('Validation normalized L1', validation_loss) 
    writer.add_scalar("Validation normalized L1", validation_loss, iter_+1)
    if best_loss > validation_loss:
        best_loss = validation_loss
        save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iter_+1) + '_best.pth'
        torch.save((model.state_dict()), save_path)
        print('Model saved to', save_path)
    else:
        pass

    scheduler.step()

    # # save checkpoint each outer epoch
    # save_path_epoch = experiment_path + experiment_name + '_E' + str((iter_+1)) + '.pth'
    # torch.save((model.state_dict()), save_path_epoch)
    # print('Model saved to', save_path_epoch)
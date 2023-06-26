import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import StepLR
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform, UnetDataTransform_norm, normalize
# Import a torch.utils.data.Dataset class that takes a list of data examples, a path to those examples
# a data transform and outputs a torch dataset.
from functions.data.mri_dataset import SliceDataset
# Unet architecture as nn.Module
from functions.models.unet import Unet
# Function that returns a MaskFunc object either for generatig random or equispaced masks
from functions.data.subsample import create_mask_for_mask_type
# Implementation of SSIMLoss
from functions.training.losses import SSIMLoss

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

####################################################################################
experiment_name = 'E6.5_maml(-3)_adapt_Q1'


# 'E6.3_maml(2inner)_adapt_Q1'
# 'E6.3_maml(2inner)_adapt_Q2'
# 'E6.3_maml(2inner)_adapt_Q3'
# 'E6.3_standard_adapt_Q1'
# 'E6.3_standard_adapt_Q2'
# 'E6.3_standard_adapt_Q3'
# 'E6.3_maml(1inner)_adapt_Q1'
# 'E6.3_maml(1inner)_adapt_Q2'
# 'E6.3_maml(1inner)_adapt_Q3'


checkpoint_path = '/cheng/metaMRI/metaMRI/save/E6.4_maml(NMSE-lre-3)_T8x200_200epoch/E6.4_maml(NMSE-lre-3)_T8x200_200epoch_E200_best.pth'
# /cheng/metaMRI/metaMRI/save/E6.3_maml(NMSE)_T8x200_2inner_200epoch/E6.3_maml(NMSE)_T8x200_2inner_200epoch_E170_best.pth
# /cheng/metaMRI/metaMRI/save/E6.3_standard(NMSE)_T8x200_100epoch/E6.3_standard(NMSE)_T8x200_100epoch_E80_best.pth
# /cheng/metaMRI/metaMRI/save/E6.3_maml(NMSE)_T8x200_1inner_200epoch/E6.3_maml(NMSE)_T8x200_1inner_200epoch_E178_best.pth

path_adapt = '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_train_AXT1POST_TrioTim_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_train_AXT1POST_TrioTim_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_train_AXFLAIR_Skyra_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_train_AXT2_Aera_5-8.yaml'

path_test = '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_test_AXT1POST_TrioTim_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_test_AXT1POST_TrioTim_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_test_AXFLAIR_Skyra_5-8.yaml'
# '/cheng/metaMRI/metaMRI/data_dict/E6.2/brain_test_AXT2_Aera_5-8.yaml'
####################################################################################

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
TRAINING_EPOCH = 30
adapt_shot = 10

# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)


data_transform = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=True, mode='adapt')

adapt_set = SliceDataset(dataset = path_adapt, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                        challenge="multicoil", transform=data_transform, use_dataset_cache=True, num_samples = adapt_shot)
print("Adapt date number: ", len(adapt_set))

test_set = SliceDataset(dataset = path_test, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                      challenge="multicoil", transform=data_transform, use_dataset_cache=True)
print("Validation date number: ", len(test_set))

# dataloader: batch size 1 
adapt_dataloader = torch.utils.data.DataLoader(dataset = adapt_set, batch_size = 1, num_workers = 8, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
test_dataloader = torch.utils.data.DataLoader(dataset = test_set, batch_size = 1, num_workers = 8, 
                    shuffle = False, generator = torch.Generator().manual_seed(1))


def train(model, dataloader, optimizer): 
    model.train()
    train_loss = 0.0
    mse_fct = torch.nn.MSELoss(reduction='sum')

    for iter, batch in tqdm(enumerate(dataloader)):
        input_image, target_image, mean, std, fname, slice_num = batch
        train_inputs = input_image.to(device)
        train_targets = target_image.to(device)
        std = std.to(device)
        mean = mean.to(device)

        train_outputs = model(train_inputs)
        train_outputs = train_outputs * std + mean

        loss = mse_fct(train_outputs, train_targets) / torch.sum(torch.abs(train_targets)**2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss


def evaluate(model, dataloader):
    model.eval()
    total_l1_loss = 0.0
    total_ssim_loss = 0.0
    total_psnr_loss = 0.0
    total_nmse_loss = 0.0
    ssim_fct = SSIMLoss()
    psner_mse_fct = torch.nn.MSELoss(reduction='mean')
    mse_fct = torch.nn.MSELoss(reduction='sum')

    for iter, batch in enumerate(dataloader): 
        input_image, target_image, mean, std, fname, slice_num = batch
        val_inputs = input_image.to(device)
        val_targets = target_image.to(device)
        std = std.to(device)
        mean = mean.to(device)

        val_outputs = model(val_inputs)
        val_outputs = val_outputs * std + mean

        # NMAE
        l1 = l1_loss(val_outputs, val_targets) / torch.sum(torch.abs(val_targets))
        total_l1_loss += l1.item()
        # NMSE 
        nmse_loss = mse_fct(val_outputs, val_targets) / torch.sum(torch.abs(val_targets)**2)
        total_nmse_loss += nmse_loss.item()
        # PSNR
        psnr_loss = 20*torch.log10(torch.tensor(val_targets.max().unsqueeze(0).item()))-10*torch.log10(psner_mse_fct(val_outputs,val_targets))
        total_psnr_loss += psnr_loss.item()
        # SSIM = 1 - loss
        ssim_loss = ssim_fct(val_outputs, val_targets, data_range = val_targets.max().unsqueeze(0))
        total_ssim_loss += (1-ssim_loss.item())

    validation_loss_l1 = total_l1_loss / len(dataloader) 
    validation_loss_NMSE = total_nmse_loss / len(dataloader)
    validation_loss_PSNR = total_psnr_loss / len(dataloader)
    validation_loss_SSIM = total_ssim_loss / len(dataloader)

    return validation_loss_l1, validation_loss_NMSE, validation_loss_PSNR, validation_loss_SSIM


model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
model.load_state_dict(torch.load(checkpoint_path))
model = model.to(device)


##########################
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
l1_loss = torch.nn.L1Loss(reduction='sum')

# best_loss = 10.000
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss = train(model, adapt_dataloader, optimizer)
    print('Adapt NMSE', training_loss) 
    writer.add_scalar("Adapt NMSE", training_loss, iteration+1)
    # val
    validation_loss_l1, validation_loss_NMSE, validation_loss_PSNR, validation_loss_SSIM = evaluate(model, test_dataloader)
    print('Testing NMAE', validation_loss_l1) 
    writer.add_scalar("Testing NMAE", validation_loss_l1, iteration+1)
    print('Testing NMSE', validation_loss_NMSE) 
    writer.add_scalar("Testing NMSE", validation_loss_NMSE, iteration+1)
    print('Testing PSNR', validation_loss_PSNR) 
    writer.add_scalar("Testing PSNR", validation_loss_PSNR, iteration+1)
    print('Testing SSIM', validation_loss_SSIM) 
    writer.add_scalar("Testing SSIM", validation_loss_SSIM, iteration+1)
    # if best_loss > validation_loss:
    #     best_loss = validation_loss
    #     save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '_best.pth'
    #     torch.save((model.state_dict()), save_path)
    #     print('Model saved to', save_path)
    # else:
    #     pass
    # #scheduler.step()

### save model
# save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '.pth'
# torch.save((model.state_dict()), save_path)
# print('Model saved to', save_path)

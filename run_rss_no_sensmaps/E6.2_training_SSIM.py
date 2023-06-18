import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
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

#experiment_name = 'E6.1_train_SSIM_kneePD_Skyra'
#experiment_name = 'E6.1_train_SSIM_kneePDFS_Aera'
#experiment_name = 'E6.1_train_SSIM_brainT1POST_TrioTim'
#experiment_name = 'E6.1_train_SSIM_knee(PD_Skyra+PDFS_Aera)'
experiment_name = 'E6.1_train_SSIM_(P1+P2+P3)'
#experiment_name = 'E6.1_train_SSIM_kneePDFS_paper'

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
TRAINING_EPOCH = 70
num_sample_train = 300
num_sample_val = 100
#path_train = '/cheng/metaMRI/metaMRI/data_dict/Acquisition/knee_train.yaml'
path_train1 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/knee_train_PD_Skyra_15-22.yaml'
path_train2 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/knee_train_PDFS_Aera_15-22.yaml'
path_train3 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/brain_train_T1POST_TrioTim5-8.yaml'
#path_val = '/cheng/metaMRI/metaMRI/data_dict/Acquisition/knee_val_CORPDFS.yaml'
path_val1 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/knee_val_PD_Skyra_15-22.yaml'
path_val2 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/knee_val_PDFS_Aera_15-22.yaml'
path_val3 = '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/E6.1/brain_val_T1POST_TrioTim5-8.yaml'

# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=False, mode='train')
data_transform_val = UnetDataTransform_norm('multicoil', mask_func = mask_function, use_seed=True, mode='val')

# dataset: num_sample_subset x 3
trainset1 = SliceDataset(dataset = path_train1, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                        challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_sample_train)
trainset2 = SliceDataset(dataset = path_train2, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                        challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_sample_train)
trainset3 = SliceDataset(dataset = path_train3, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                        challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_sample_train)
trainset = torch.utils.data.ConcatDataset([trainset1, trainset2, trainset3])

print("Training date number: ", len(trainset))
valset1 = SliceDataset(dataset = path_val1, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                      challenge="multicoil", transform=data_transform_val, use_dataset_cache=True, num_samples= num_sample_val)
valset2 = SliceDataset(dataset = path_val2, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                      challenge="multicoil", transform=data_transform_val, use_dataset_cache=True, num_samples= num_sample_val)
valset3 = SliceDataset(dataset = path_val3, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                      challenge="multicoil", transform=data_transform_val, use_dataset_cache=True, num_samples= num_sample_val)
valset = torch.utils.data.ConcatDataset([valset1, valset2, valset3])
print("Validation date number: ", len(valset))

# dataloader: batch size 1 
train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 8, num_workers = 8, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
val_dataloader = torch.utils.data.DataLoader(dataset = valset, batch_size = 1, num_workers = 8, 
                    shuffle = False, generator = torch.Generator().manual_seed(1))


def train(model, dataloader, optimizer): 
    model.train()
    train_loss = 0.0
    ssim_fct = SSIMLoss()

    for iter, batch in tqdm(enumerate(dataloader)):
        input_image, target_image, mean, std, fname, slice_num = batch
        train_inputs = input_image.to(device)
        train_targets = target_image.to(device)
        std = std.to(device)
        mean = mean.to(device)

        train_outputs = model(train_inputs)
        train_outputs = train_outputs * std + mean

        # SSIM = 1 - loss
        loss = ssim_fct(train_outputs, train_targets, data_range = train_targets.max().unsqueeze(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += (1-loss.item())

    avg_train_loss = train_loss / len(dataloader)
    return avg_train_loss


def evaluate(model, dataloader):
    model.eval()
    total_val_loss = 0.0
    ssim_fct = SSIMLoss()
    
    for iter, batch in tqdm(enumerate(dataloader)): 
        input_image, target_image, mean, std, fname, slice_num = batch
        val_inputs = input_image.to(device)
        val_targets = target_image.to(device)
        std = std.to(device)
        mean = mean.to(device)

        val_outputs = model(val_inputs)
        val_outputs = val_outputs * std + mean

        # SSIM = 1 - loss
        loss = ssim_fct(val_outputs, val_targets, data_range = val_targets.max().unsqueeze(0))

        total_val_loss += (1-loss.item())
        
    validation_loss = total_val_loss / len(dataloader)
    return validation_loss


model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
model = model.to(device)


##########################
optimizer = torch.optim.Adam(model.parameters(),lr=0.001)
# optimizer = torch.optim.RMSprop(model.parameters(),lr=0.0001,weight_decay=0.0)
#scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8)
best_loss = 0.000
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss = train(model, train_dataloader, optimizer)
    print('Training SSIM', training_loss) 
    writer.add_scalar("Training SSIM", training_loss, iteration+1)
    # val
    validation_loss = evaluate(model, val_dataloader)
    print('Validation SSIM', validation_loss) 
    writer.add_scalar("Validation SSIM", validation_loss, iteration+1)
    if best_loss < validation_loss:
        best_loss = validation_loss
        save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '_best.pth'
        torch.save((model.state_dict()), save_path)
        print('Model saved to', save_path)
    else:
        pass
    #scheduler.step()

### save model
save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '.pth'
torch.save((model.state_dict()), save_path)
print('Model saved to', save_path)

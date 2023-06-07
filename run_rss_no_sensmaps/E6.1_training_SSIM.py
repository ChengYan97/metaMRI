import random
import numpy as np
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import learn2learn as l2l
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
# The corase reconstruction is the rss of the zerofilled multi-coil kspaces
# after inverse FT.
from functions.data.transforms import UnetDataTransform, normalize
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

#experiment_name = 'E6.1_train_SSIM_kneePDFS_Aera_14-22'
experiment_name = 'E6.1_train_SSIM_kneePD'

# tensorboard dir
experiment_path = '/cheng/metaMRI/metaMRI/save/' + experiment_name + '/'
writer = SummaryWriter(experiment_path)

# seed
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)

# hyperparameter
TRAINING_EPOCH = 70
num_sample_train = 300
num_sample_val = 300
path_train = '/cheng/metaMRI/metaMRI/data_dict/Acquisition/knee_train_CORPD.yaml'
# '/cheng/metaMRI/metaMRI/data_dict_narrowSlices/knee_train_PDFS_Aera.yaml'
path_val = '/cheng/metaMRI/metaMRI/data_dict/Acquisition/knee_val_CORPD.yaml'
#'/cheng/metaMRI/metaMRI/data_dict_narrowSlices/knee_val_PDFS_Aera.yaml'

# mask function and data transform
mask_function = create_mask_for_mask_type(mask_type_str = 'random', self_sup = False, 
                    center_fraction = 0.08, acceleration = 4.0, acceleration_total = 3.0)

data_transform_train = UnetDataTransform('multicoil', mask_func = mask_function, use_seed = False)
data_transform_val = UnetDataTransform('multicoil', mask_func = mask_function, use_seed=True)

# dataset: num_sample_subset x 3
trainset = SliceDataset(dataset = path_train, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                        challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_sample_train)

valset = SliceDataset(dataset = path_val, path_to_dataset='', path_to_sensmaps=None, provide_senmaps=False, 
                      challenge="multicoil", transform=data_transform_train, use_dataset_cache=True, num_samples= num_sample_val)

# dataloader: batch size 1 
train_dataloader = torch.utils.data.DataLoader(dataset = trainset, batch_size = 1, num_workers = 8, 
                    shuffle = True, generator = torch.Generator().manual_seed(1), pin_memory = True)
test_dataloader = torch.utils.data.DataLoader(dataset = valset, batch_size = 1, num_workers = 8, 
                    shuffle = False, generator = torch.Generator().manual_seed(1), pin_memory = True)


def train(model, dataloader, optimizer): 
    model.train()
    train_loss = 0.0
    ssim_fct = SSIMLoss()

    for iter, batch in enumerate(dataloader):
        train_inputs = batch.image.unsqueeze(1).to(device)
        train_targets = batch.target.unsqueeze(1).to(device)

        train_outputs = model(train_inputs)

        # SSIM = 1 - loss
        loss = ssim_fct(train_outputs, train_targets, data_range = train_targets.max().unsqueeze(0))
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += (1-loss.item())

    avg_train_loss = train_loss / len(dataloader.dataset)
    return avg_train_loss


def evaluate(model, dataloader):
    model.eval()
    total_val_loss = 0.0
    ssim_fct = SSIMLoss()
    
    for iter, batch in enumerate(dataloader): 
        val_inputs = batch.image.unsqueeze(1).to(device)
        val_targets = batch.target.unsqueeze(1).to(device)

        val_outputs = model(val_inputs)

        # SSIM = 1 - loss
        loss = ssim_fct(val_outputs, val_targets, data_range = val_targets.max().unsqueeze(0))

        total_val_loss += (1-loss.item())
        
    validation_loss = total_val_loss / len(dataloader.dataset)
    return validation_loss


model = Unet(in_chans=1, out_chans=1, chans=32, num_pool_layers=4, drop_prob=0.0)
model = model.to(device)


##########################
# optimizer = torch.optim.Adam(model.parameters(),lr=0.001, betas=(0.9, 0.999), 
#                              eps=1e-08, weight_decay=0.0, amsgrad=False)
optimizer = torch.optim.Adam(model.parameters(),lr=0.0001)
# optimizer = torch.optim.RMSprop(model.parameters(),lr=0.0001,weight_decay=0.0)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5, eta_min=1e-8)
best_loss = 0
for iteration in range(TRAINING_EPOCH):
    print('Iteration:', iteration+1)
    # training
    training_loss = train(model, train_dataloader, optimizer)
    print('Training SSIM', training_loss) 
    writer.add_scalar("Training SSIM", training_loss, iteration)
    # val
    validation_loss = evaluate(model, test_dataloader)
    print('Validation SSIM', validation_loss) 
    writer.add_scalar("Validation SSIM", validation_loss, iteration)
    if best_loss < validation_loss:
        best_loss = validation_loss
        save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '_E' + str(iteration+1) + '_best.pth'
        torch.save((model.state_dict()), save_path)
        print('Model saved to', save_path)
    else:
        pass
    #scheduler.step(val_loss_history[-1])

### save model
save_path = '/cheng/metaMRI/metaMRI/save/'+ experiment_name + '/' + experiment_name + '.pth'
torch.save((model.state_dict()), save_path)
print('Model saved to', save_path)
